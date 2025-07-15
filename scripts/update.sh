#!/bin/bash

COMMIT_MSG="${1:-$(date +"%Y-%m-%d %H:%M:%S")}"
# git pull https://Frank2002code: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@github.com/Frank2002code/DRL-on-parameter-extraction.git main
git pull
git status
git add -A
git commit -m "$COMMIT_MSG" || echo "No changes to commit"
# git push https://Frank2002code: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@github.com/Frank2002code/DRL-on-parameter-extraction.git main
git push