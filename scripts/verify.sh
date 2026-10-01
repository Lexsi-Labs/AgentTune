#!/usr/bin/env bash
# verify.sh — re-check the spine's green state by hand. No CI, no automation.
#
#   ./scripts/verify.sh          # run everything
#   ./scripts/verify.sh studies  # only the case studies
#   ./scripts/verify.sh docs     # only the strict docs build
#
# Checks:
#   1. Every case study under examples/case_studies/ runs clean, GPU-free.
#   2. `mkdocs build --strict` produces the site with zero warnings.
#      (needs the docs extra:  pip install 'agenttune[docs]')
#
# Exit code is non-zero if anything fails, so you can eyeball the summary at the end.

set -u
cd "$(dirname "$0")/.." || exit 2
WHAT="${1:-all}"
studies_rc=0
docs_rc=0

run_studies() {
  echo "== case studies =="
  local pass=0 fail=0 f
  for f in $(find examples/case_studies -name '*.py' | sort); do
    if python "$f" >/dev/null 2>&1; then
      pass=$((pass + 1))
    else
      fail=$((fail + 1))
      echo "  FAIL  $f"
    fi
  done
  echo "  -> ${pass} passed, ${fail} failed"
  [ "$fail" -eq 0 ] || studies_rc=1
}

run_docs() {
  echo "== docs (mkdocs build --strict) =="
  if ! python -c "import mkdocs" 2>/dev/null; then
    echo "  SKIP  mkdocs not installed (pip install 'agenttune[docs]')"
    return
  fi
  local out
  out="$(python -m mkdocs build --strict -d /tmp/agenttune-verify-site 2>&1)"
  if [ $? -eq 0 ]; then
    echo "  -> built clean ($(find /tmp/agenttune-verify-site -name '*.html' | wc -l | tr -d ' ') pages, 0 warnings)"
  else
    echo "$out" | grep -E "WARNING|ERROR" | sed 's/^/  /'
    echo "  -> FAILED"
    docs_rc=1
  fi
  rm -rf /tmp/agenttune-verify-site
}

case "$WHAT" in
  studies) run_studies ;;
  docs)    run_docs ;;
  all)     run_studies; run_docs ;;
  *) echo "usage: $0 [all|studies|docs]"; exit 2 ;;
esac

echo
if [ "$studies_rc" -eq 0 ] && [ "$docs_rc" -eq 0 ]; then
  echo "ALL GREEN"
  exit 0
else
  echo "NOT GREEN — see failures above"
  exit 1
fi
