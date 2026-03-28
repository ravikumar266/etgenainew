"""
agent/tools_github.py
─────────────────────
GitHub code review and repository tools using the GitHub REST API v3.

User-facing @tools:
  review_pr(repo, pr_number)              → full AI code review of a PR
  get_pr_files(repo, pr_number)           → list files changed in a PR with diffs
  list_prs(repo, state)                   → list open/closed/all PRs in a repo
  get_file(repo, path, branch)            → read any file from a repository
  search_code(repo, query)                → search code within a repository

Required .env:
  GITHUB_TOKEN = ghp_xxxxxxxxxxxxxxxxxxxx
  Get one at: github.com/settings/tokens → New token (classic)
  Required scopes: repo (for private repos) or public_repo (for public only)

Install:
  No new packages needed — uses requests (already installed)
"""

import os
import re
from typing import Optional

import requests
from langchain_core.tools import tool

from agent.config import llm, logger

# ── GitHub API client ─────────────────────────────────────────────────────────

_GITHUB_API = "https://api.github.com"
_GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")


def _gh_headers() -> dict:
    """Build GitHub API request headers."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if _GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {_GITHUB_TOKEN}"
    return headers


def _gh_get(path: str, params: Optional[dict] = None) -> tuple[dict | list, int]:
    """
    Make a GET request to the GitHub API.
    Returns (data, status_code).
    """
    url = f"{_GITHUB_API}/{path.lstrip('/')}"
    try:
        resp = requests.get(url, headers=_gh_headers(), params=params, timeout=15)
        try:
            data = resp.json()
        except Exception:
            data = {"message": resp.text}
        return data, resp.status_code
    except requests.exceptions.Timeout:
        return {"message": "GitHub API request timed out"}, 408
    except Exception as e:
        return {"message": str(e)}, 500


def _parse_repo(repo: str) -> str:
    """
    Normalise repo input to 'owner/repo' format.
    Accepts:
      - 'owner/repo'
      - 'https://github.com/owner/repo'
      - 'github.com/owner/repo'
    """
    repo = repo.strip().rstrip("/")
    match = re.search(r"github\.com[/:]([^/]+/[^/]+?)(?:\.git)?$", repo)
    if match:
        return match.group(1)
    if repo.count("/") == 1:
        return repo
    return repo


def _format_error(data: dict, status: int, context: str) -> str:
    """Return a readable error string from a GitHub API error response."""
    msg = data.get("message", "Unknown error")
    if status == 401:
        return f"❌ Authentication failed. Check your GITHUB_TOKEN in .env. ({msg})"
    if status == 403:
        return f"❌ Access denied. Your token may lack required scopes. ({msg})"
    if status == 404:
        return f"❌ Not found: {context}. Check the repo name and PR number. ({msg})"
    if status == 422:
        return f"❌ Validation error: {msg}"
    return f"❌ GitHub API error {status}: {msg}"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_pr_diff(repo: str, pr_number: int) -> str:
    """Fetch the unified diff for a PR (raw text, not JSON)."""
    url = f"{_GITHUB_API}/repos/{repo}/pulls/{pr_number}"
    headers = {**_gh_headers(), "Accept": "application/vnd.github.diff"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code == 200:
            return resp.text
        return ""
    except Exception:
        return ""


def _truncate_diff(diff: str, max_chars: int = 12000) -> str:
    """
    Truncate a large diff to fit within the LLM context window.
    Preserves the beginning (which has the most important changes) and
    adds a note if truncated.
    """
    if len(diff) <= max_chars:
        return diff
    truncated = diff[:max_chars]
    # Try to cut at a clean line boundary
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.8:
        truncated = truncated[:last_newline]
    remaining = len(diff) - len(truncated)
    return truncated + f"\n\n... [{remaining} chars truncated — diff too large for full review]"


def _ai_review(repo: str, pr_title: str, pr_body: str, diff: str, files_summary: str) -> str:
    """
    Send the PR diff to Gemini for a structured code review.
    Returns the review text.
    """
    prompt = f"""You are an expert code reviewer. Perform a thorough, constructive review of this GitHub Pull Request.

REPOSITORY: {repo}
PR TITLE: {pr_title}
PR DESCRIPTION: {pr_body or "(no description provided)"}

FILES CHANGED:
{files_summary}

DIFF:
{diff}

Provide a structured review with these sections:

## Summary
Brief overview of what this PR does and your overall assessment.

## Issues Found
For each issue, specify:
- **Severity**: Critical / High / Medium / Low
- **File**: filename and line number if applicable
- **Issue**: Clear description of the problem
- **Fix**: Concrete suggestion for how to fix it

If no issues found in a severity category, skip it.

## Security Concerns
Any security vulnerabilities, exposed secrets, or risky patterns.

## Code Quality
Comments on readability, maintainability, naming, structure.

## Positive Highlights
Things done well — good patterns, clean code, good test coverage.

## Recommendations
Top 3 action items the author should address before merging.

Be specific and actionable. Reference exact file names and line numbers where possible.
"""
    try:
        response = llm.invoke(prompt)
        # Extract text from Gemini response (handles both str and list content)
        content = response.content
        if isinstance(content, list):
            return " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
        return str(content).strip()
    except Exception as e:
        return f"AI review failed: {str(e)}"


# ── User-facing @tools ────────────────────────────────────────────────────────

@tool
def review_pr(repo: str, pr_number: int) -> str:
    """
    Perform a full AI-powered code review of a GitHub Pull Request.

    Fetches the PR diff and all changed files, then uses the LLM to produce
    a structured review covering bugs, security issues, code quality,
    and concrete fix suggestions.

    Args:
      repo      : Repository in 'owner/repo' format, or full GitHub URL.
                  Examples: "torvalds/linux", "https://github.com/owner/repo"
      pr_number : The pull request number (the number after /pull/ in the URL).

    Returns a structured review with severity-ranked issues and recommendations.
    Requires GITHUB_TOKEN in .env for private repos.
    """
    if not _GITHUB_TOKEN:
        logger.warning("[GitHub] GITHUB_TOKEN not set — requests will be rate-limited (60/hour)")

    repo = _parse_repo(repo)

    # Step 1: Fetch PR metadata
    pr_data, status = _gh_get(f"repos/{repo}/pulls/{pr_number}")
    if status != 200:
        return _format_error(pr_data, status, f"PR #{pr_number} in {repo}")

    pr_title = pr_data.get("title", "(untitled)")
    pr_body = pr_data.get("body", "") or ""
    pr_state = pr_data.get("state", "unknown")
    pr_author = pr_data.get("user", {}).get("login", "unknown")
    pr_base = pr_data.get("base", {}).get("ref", "main")
    pr_head = pr_data.get("head", {}).get("ref", "unknown")
    pr_additions = pr_data.get("additions", 0)
    pr_deletions = pr_data.get("deletions", 0)
    pr_changed_files = pr_data.get("changed_files", 0)
    pr_url = pr_data.get("html_url", "")

    logger.info(f"[GitHub] Reviewing PR #{pr_number} in {repo}: '{pr_title}'")

    # Step 2: Fetch list of changed files
    files_data, fstatus = _gh_get(f"repos/{repo}/pulls/{pr_number}/files", {"per_page": 50})
    files_summary = ""
    if fstatus == 200 and isinstance(files_data, list):
        file_lines = []
        for f in files_data[:20]:  # cap at 20 files for summary
            fname = f.get("filename", "")
            status_str = f.get("status", "")
            adds = f.get("additions", 0)
            dels = f.get("deletions", 0)
            file_lines.append(f"  {status_str.upper():10s} +{adds}/-{dels}  {fname}")
        files_summary = "\n".join(file_lines)
        if len(files_data) > 20:
            files_summary += f"\n  ... and {len(files_data) - 20} more files"
    else:
        files_summary = "(could not fetch file list)"

    # Step 3: Fetch the unified diff
    diff = _get_pr_diff(repo, pr_number)
    if not diff:
        diff = "(diff not available — the PR may be too large or access is restricted)"

    diff = _truncate_diff(diff, max_chars=12000)

    # Step 4: Run AI review
    review = _ai_review(repo, pr_title, pr_body, diff, files_summary)

    # Step 5: Build final output
    header = (
        f"# Code Review: PR #{pr_number}\n"
        f"**Repo:** {repo}  |  **URL:** {pr_url}\n"
        f"**Title:** {pr_title}\n"
        f"**Author:** {pr_author}  |  **State:** {pr_state}\n"
        f"**Branch:** {pr_head} → {pr_base}\n"
        f"**Changes:** +{pr_additions} / -{pr_deletions} across {pr_changed_files} file(s)\n\n"
        f"---\n\n"
    )

    return header + review


@tool
def get_pr_files(repo: str, pr_number: int) -> str:
    """
    List all files changed in a GitHub Pull Request with their diffs.

    Args:
      repo      : Repository in 'owner/repo' format or full GitHub URL.
      pr_number : The pull request number.

    Returns each file's status (added/modified/deleted), line changes, and patch diff.
    """
    repo = _parse_repo(repo)
    data, status = _gh_get(f"repos/{repo}/pulls/{pr_number}/files", {"per_page": 100})

    if status != 200:
        return _format_error(data, status, f"PR #{pr_number} files in {repo}")

    if not isinstance(data, list) or not data:
        return f"No files found in PR #{pr_number} of {repo}."

    lines = [f"Files changed in PR #{pr_number} ({repo}) — {len(data)} file(s):\n"]

    for f in data:
        fname = f.get("filename", "unknown")
        fstatus = f.get("status", "unknown")
        adds = f.get("additions", 0)
        dels = f.get("deletions", 0)
        patch = f.get("patch", "")

        lines.append(f"\n{'─'*60}")
        lines.append(f"File:    {fname}")
        lines.append(f"Status:  {fstatus}  |  +{adds} / -{dels}")

        if patch:
            # Truncate very long patches
            if len(patch) > 3000:
                patch = patch[:3000] + "\n... [patch truncated]"
            lines.append(f"Patch:\n{patch}")
        else:
            lines.append("Patch:   (binary file or no diff available)")

    return "\n".join(lines)


@tool
def list_prs(repo: str, state: str = "open") -> str:
    """
    List pull requests in a GitHub repository.

    Args:
      repo  : Repository in 'owner/repo' format or full GitHub URL.
      state : Filter by state — "open" (default), "closed", or "all".

    Returns PR number, title, author, branch info, and URL for each PR.
    """
    repo = _parse_repo(repo)
    state = state.lower().strip()
    if state not in ("open", "closed", "all"):
        state = "open"

    data, status = _gh_get(
        f"repos/{repo}/pulls",
        {"state": state, "per_page": 20, "sort": "updated", "direction": "desc"}
    )

    if status != 200:
        return _format_error(data, status, f"PRs in {repo}")

    if not isinstance(data, list) or not data:
        return f"No {state} pull requests found in {repo}."

    lines = [f"Pull requests in {repo} (state: {state}) — showing {len(data)}:\n"]

    for pr in data:
        number = pr.get("number", "?")
        title = pr.get("title", "(untitled)")
        author = pr.get("user", {}).get("login", "unknown")
        pr_state = pr.get("state", "unknown")
        base = pr.get("base", {}).get("ref", "?")
        head = pr.get("head", {}).get("ref", "?")
        url = pr.get("html_url", "")
        draft = " [DRAFT]" if pr.get("draft") else ""
        updated = pr.get("updated_at", "")[:10]  # date only

        lines.append(
            f"\nPR #{number}{draft} — {title}\n"
            f"  Author: {author}  |  {head} → {base}  |  {pr_state}  |  updated {updated}\n"
            f"  URL: {url}"
        )

    return "\n".join(lines)


@tool
def get_file(repo: str, path: str, branch: str = "main") -> str:
    """
    Read the content of any file from a GitHub repository.

    Args:
      repo   : Repository in 'owner/repo' format or full GitHub URL.
      path   : File path within the repo (e.g. "src/main.py", "README.md").
      branch : Branch name to read from (default: "main"). Also accepts "master".

    Returns the file content as plain text.
    Useful for reading specific files mentioned in a PR review.
    """
    import base64

    repo = _parse_repo(repo)
    data, status = _gh_get(
        f"repos/{repo}/contents/{path.lstrip('/')}",
        {"ref": branch}
    )

    if status != 200:
        # Try 'master' if 'main' fails
        if branch == "main" and status == 404:
            data, status = _gh_get(
                f"repos/{repo}/contents/{path.lstrip('/')}",
                {"ref": "master"}
            )
            if status != 200:
                return _format_error(data, status, f"{path} in {repo}@{branch}")
        else:
            return _format_error(data, status, f"{path} in {repo}@{branch}")

    if isinstance(data, list):
        # Path is a directory, not a file
        names = [item.get("name", "") for item in data]
        return f"'{path}' is a directory in {repo}. Contents:\n" + "\n".join(f"  {n}" for n in names)

    encoding = data.get("encoding", "")
    content_b64 = data.get("content", "")
    size = data.get("size", 0)
    file_url = data.get("html_url", "")

    if size > 500_000:
        return f"File '{path}' is {size:,} bytes — too large to display. View it at: {file_url}"

    if encoding == "base64" and content_b64:
        try:
            content = base64.b64decode(content_b64).decode("utf-8", errors="replace")
            # Truncate very large files
            if len(content) > 15000:
                content = content[:15000] + f"\n\n... [{len(content)-15000} chars truncated]"
            return (
                f"File: {path}  |  Repo: {repo}  |  Branch: {branch}\n"
                f"URL: {file_url}\n"
                f"{'─'*60}\n"
                f"{content}"
            )
        except Exception as e:
            return f"Could not decode file content: {str(e)}"

    return f"File '{path}' has unsupported encoding '{encoding}'. View at: {file_url}"


@tool
def search_code(repo: str, query: str) -> str:
    """
    Search for code within a specific GitHub repository.

    Args:
      repo  : Repository in 'owner/repo' format or full GitHub URL.
      query : Search query (e.g. "def authenticate", "TODO", "password", "import requests").

    Returns matching files with line excerpts showing where the query appears.
    Useful for finding where a function is defined, spotting security patterns,
    or locating all usages of a variable across the codebase.

    Note: GitHub code search requires authentication (GITHUB_TOKEN in .env).
    """
    if not _GITHUB_TOKEN:
        return (
            "❌ Code search requires a GITHUB_TOKEN. "
            "Set it in .env and restart. "
            "Get one at github.com/settings/tokens"
        )

    repo = _parse_repo(repo)
    search_query = f"{query} repo:{repo}"

    data, status = _gh_get(
        "search/code",
        {"q": search_query, "per_page": 10}
    )

    if status == 422:
        return "Search query is invalid or too short. Try a more specific term."
    if status != 200:
        return _format_error(data, status, f"code search in {repo}")

    items = data.get("items", [])
    total = data.get("total_count", 0)

    if not items:
        return f"No results found for '{query}' in {repo}."

    lines = [f"Code search: '{query}' in {repo} — {total} total match(es), showing {len(items)}:\n"]

    for item in items:
        fname = item.get("path", "unknown")
        url = item.get("html_url", "")
        # text_matches available when Accept header includes text-match
        lines.append(f"\n  {fname}\n  {url}")

    return "\n".join(lines)