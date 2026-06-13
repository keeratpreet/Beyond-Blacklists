#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# PUSH_TO_GITHUB.sh
# Run this ONCE from your Mac Terminal to push "Beyond Blacklists" to GitHub.
#
# BEFORE RUNNING:
#   1. Create an empty repo on GitHub called "Beyond-Blacklists" (no README)
#   2. Replace YOUR_GITHUB_USERNAME below with your actual GitHub username
#   3. Have your Personal Access Token ready (you'll be prompted for it)
# ─────────────────────────────────────────────────────────────────────────────

GITHUB_USERNAME="keeratpreet"
REPO_NAME="Beyond-Blacklists"
REMOTE_URL="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

set -e
cd "$(dirname "$0")"

echo ""
echo "=== Step 1: Installing Git LFS (via Homebrew) ==="
if ! command -v git-lfs &>/dev/null; then
  brew install git-lfs
else
  echo "git-lfs already installed: $(git lfs version)"
fi

echo ""
echo "=== Step 2: Initialising Git LFS ==="
git lfs install

echo ""
echo "=== Step 3: Renaming branch to main ==="
git branch -m main 2>/dev/null || true

echo ""
echo "=== Step 4: Staging all files ==="
git add .
git status --short | head -40
echo "..."
echo "Total files staged: $(git diff --cached --name-only | wc -l | xargs)"

echo ""
echo "=== Step 5: Committing ==="
git commit -m "Initial commit: Beyond Blacklists — YOLOv8 + URL analysis phishing detector

- YOLOv8 visual branch (CSPDarknet53 + PANet) for logo detection
- BiT-M-R50x1 Siamese network (2048-dim embeddings) for brand matching
- tldextract domain consistency check
- Heuristic + Random Forest URL analysis (36 features, 125k training URLs)
- FastAPI server with workspace auto-detection
- Chrome extension (Manifest V3) with fixed domain verification logic
- Full training, evaluation and scoring scripts"

echo ""
echo "=== Step 6: Adding remote ==="
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"
echo "Remote set to: $REMOTE_URL"

echo ""
echo "=== Step 7: Pushing (you will be prompted for your GitHub token) ==="
echo "Username: enter your GitHub username"
echo "Password: paste your Personal Access Token (NOT your password)"
echo ""
git push -u origin main

echo ""
echo "✅  Done! Your repo is live at: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
