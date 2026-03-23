"""
Slack Bot — handles /deploy and /deploy-status slash commands.

/deploy username/repo branch   → triggers GitHub Actions workflow_dispatch
/deploy-status                 → shows latest deployment status

Run locally:
    pip install flask slack-sdk requests
    python bot.py

Deploy to Cloud Run:
    docker build -t slack-bot . && docker run -p 3000:3000 slack-bot
"""

import os
import hmac
import hashlib
import time
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Config (set these as environment variables) ──────────────────────────────
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
GITHUB_TOKEN         = os.environ["GITHUB_TOKEN"]
SLACK_BOT_TOKEN      = os.environ["SLACK_BOT_TOKEN"]

GITHUB_API = "https://api.github.com"
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


# ── Slack request verification ────────────────────────────────────────────────

def verify_slack_signature(req: request) -> bool:
    """Reject requests that didn't come from Slack."""
    timestamp = req.headers.get("X-Slack-Request-Timestamp", "")
    if abs(time.time() - int(timestamp)) > 60 * 5:
        return False  # replay attack guard

    sig_base = f"v0:{timestamp}:{req.get_data(as_text=True)}"
    expected = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(),
        sig_base.encode(),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, req.headers.get("X-Slack-Signature", ""))


# ── Slack helpers ─────────────────────────────────────────────────────────────

def post_message(channel: str, blocks: list, text: str = ""):
    """Post a rich Block Kit message to a Slack channel."""
    requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
        json={"channel": channel, "blocks": blocks, "text": text},
    )


def error_response(msg: str):
    return jsonify({"response_type": "ephemeral", "text": f"❌ {msg}"})


# ── GitHub helpers ────────────────────────────────────────────────────────────

def trigger_workflow(owner: str, repo: str, branch: str, environment: str = "dev"):
    """Trigger workflow_dispatch on the deploy.yml workflow."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/workflows/deploy.yml/dispatches"
    resp = requests.post(
        url,
        headers=HEADERS,
        json={"ref": branch, "inputs": {"environment": environment}},
    )
    return resp.status_code == 204  # 204 = accepted, no body


def get_latest_run(owner: str, repo: str):
    """Fetch the most recent workflow run for deploy.yml."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/workflows/deploy.yml/runs"
    resp = requests.get(url, headers=HEADERS, params={"per_page": 1})
    if resp.status_code != 200:
        return None
    runs = resp.json().get("workflow_runs", [])
    return runs[0] if runs else None


# ── /deploy command ───────────────────────────────────────────────────────────

@app.route("/slack/deploy", methods=["POST"])
def handle_deploy():
    if not verify_slack_signature(request):
        return error_response("Invalid Slack signature."), 401

    text    = request.form.get("text", "").strip()
    user    = request.form.get("user_name", "unknown")
    channel = request.form.get("channel_id", "")

    # Parse: /deploy username/repo branch
    parts = text.split()
    if len(parts) < 2 or "/" not in parts[0]:
        return error_response(
            "Usage: `/deploy username/repo_name branch`\n"
            "Example: `/deploy Rams0510/ml-deploy main`"
        )

    repo_full = parts[0]          # e.g. Rams0510/ml-deploy
    branch    = parts[1]          # e.g. main
    owner, repo = repo_full.split("/", 1)

    # Determine environment from branch name
    env_map = {"main": "dev", "staging": "staging", "prod": "prod"}
    environment = env_map.get(branch, "dev")

    # Trigger the workflow
    success = trigger_workflow(owner, repo, branch, environment)
    if not success:
        return error_response(
            f"Failed to trigger workflow for `{repo_full}` on branch `{branch}`. "
            "Check that the repo exists and GITHUB_TOKEN has workflow permissions."
        )

    # Post rich notification to channel
    post_message(channel, blocks=[
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "🚀 Deployment Triggered"}
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Repo:*\n`{repo_full}`"},
                {"type": "mrkdwn", "text": f"*Branch:*\n`{branch}`"},
                {"type": "mrkdwn", "text": f"*Environment:*\n`{environment}`"},
                {"type": "mrkdwn", "text": f"*Triggered by:*\n`{user}`"},
            ]
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Watch Run"},
                    "url": f"https://github.com/{repo_full}/actions"
                }
            ]
        }
    ], text=f"Deployment triggered for {repo_full}@{branch} by {user}")

    # Immediate ephemeral ack back to the user
    return jsonify({
        "response_type": "ephemeral",
        "text": f"✅ Deploy triggered for `{repo_full}` on `{branch}`. Watch progress in <https://github.com/{repo_full}/actions|GitHub Actions>."
    })


# ── /deploy-status command ────────────────────────────────────────────────────

@app.route("/slack/deploy-status", methods=["POST"])
def handle_deploy_status():
    if not verify_slack_signature(request):
        return error_response("Invalid Slack signature."), 401

    text = request.form.get("text", "").strip()

    # Default to your repo if none provided
    repo_full = text if text and "/" in text else "Rams0510/ml-deploy"
    owner, repo = repo_full.split("/", 1)

    run = get_latest_run(owner, repo)
    if not run:
        return error_response(f"No workflow runs found for `{repo_full}`.")

    status     = run.get("status")       # queued | in_progress | completed
    conclusion = run.get("conclusion")   # success | failure | cancelled | None
    branch     = run.get("head_branch")
    commit     = run.get("head_sha", "")[:7]
    actor      = run.get("triggering_actor", {}).get("login", "unknown")
    run_url    = run.get("html_url", "")
    created_at = run.get("created_at", "")

    if status == "completed":
        icon = "✅" if conclusion == "success" else "❌"
        state_text = f"{icon} {conclusion.capitalize()}"
    elif status == "in_progress":
        state_text = "⏳ In progress"
    else:
        state_text = "🕐 Queued"

    return jsonify({
        "response_type": "in_channel",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"Deployment Status — {repo_full}"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Status:*\n{state_text}"},
                    {"type": "mrkdwn", "text": f"*Branch:*\n`{branch}`"},
                    {"type": "mrkdwn", "text": f"*Commit:*\n`{commit}`"},
                    {"type": "mrkdwn", "text": f"*Actor:*\n`{actor}`"},
                    {"type": "mrkdwn", "text": f"*Started:*\n{created_at[:16].replace('T', ' ')} UTC"},
                ]
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View Run"},
                        "url": run_url
                    }
                ]
            }
        ]
    })


# ── Health check ──────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=False)