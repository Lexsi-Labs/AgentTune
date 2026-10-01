#!/usr/bin/env bash
# Secret scanner for the agenttune repo.
#
# Scans staged files (when run as a pre-commit hook) or the whole working tree
# (when run with --all / in CI) for common credential patterns. Fails the
# commit / CI run if a real secret is found.
#
# Patterns: OpenAI-compatible provider keys (ix_), Groq (gsk_), OpenAI (sk-),
# HuggingFace (hf_), AWS (AKIA), GitHub PATs (ghp_/github_pat_).
# Placeholders (gsk_..., sk-xxx...) and env-var lookups are allowed.
#
# Install as a pre-commit hook:
#   ln -s ../../scripts/check_secrets.sh .git/hooks/pre-commit
#   chmod +x scripts/check_secrets.sh
# Run manually:
#   ./scripts/check_secrets.sh --all
set -euo pipefail

# Allow-list of placeholder/non-secret strings that match the patterns.
ALLOW_REGEX='gsk_\.\.\.|sk-x{20,}|os\.environ|os\.getenv|OPENAI_API_KEY|GROQ_API_KEY'

# Regexes for real secrets. Anchored on the token prefix + sufficient entropy.
PATTERNS=(
  'ix_[a-f0-9]{40,}'                 # hosted-endpoint API key
  'gsk_[A-Za-z0-9]{30,}'             # Groq API key
  'sk-[A-Za-z0-9]{40,}'              # OpenAI API key (sk-proj-... / sk-...)
  'hf_[A-Za-z0-9]{30,}'              # HuggingFace token
  'AKIA[A-Z0-9]{16}'                 # AWS access key id
  'ghp_[A-Za-z0-9]{36,}'             # GitHub personal access token
  'github_pat_[A-Za-z0-9_]{20,}'     # GitHub fine-grained PAT
)

# Decide what to scan: staged files (--cached) by default, or all tracked files
# with --all. (Avoid bash 4+ `mapfile` so this runs on macOS default bash 3.2.)
if [[ "${1:-}" == "--all" ]]; then
  FILES=$(git ls-files | grep -vE '\.(png|jpg|jpeg|gif|parquet|sqlite|pkl|so|pyc)$' || true)
else
  FILES=$(git diff --cached --name-only --diff-filter=ACMR | grep -vE '\.(png|jpg|jpeg|gif|parquet|sqlite|pkl|so|pyc)$' || true)
fi

if [[ -z "$FILES" ]]; then
  exit 0
fi

found=0
for pat in "${PATTERNS[@]}"; do
  # grep -EIHn: extended regex, binary-as-text, line nums, no filename
  while IFS= read -r line; do
    # skip allow-listed placeholders / env-var references
    if echo "$line" | grep -qE "$ALLOW_REGEX"; then
      continue
    fi
    echo "SECRET SCAN: potential secret matches /$pat/"
    echo "  $line"
    found=1
  done < <(grep -EIHn "$pat" $FILES 2>/dev/null || true)
done

if [[ $found -ne 0 ]]; then
  echo ""
  echo "ERROR: secret scanner found potential hardcoded credentials."
  echo "If these are real, ROTATE the key and load it from an env var instead."
  echo "(placeholders like gsk_... and os.environ lookups are allowed.)"
  exit 1
fi
exit 0
