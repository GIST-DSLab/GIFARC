# Anonymous Submission Checklist

This repository is prepared for double-blind review through an anonymous GitHub mirror. Anonymous mirrors hide repository ownership, but they do not remove identifying text inside files. Keep the source repository review-safe before sharing the mirror URL.

## Do Not Commit

- Author names, emails, affiliations, GitHub usernames, Hugging Face usernames, or lab/server names.
- Local paths such as `/home/<user>/...`, Windows user paths, cluster paths, or Overleaf exports.
- `.env` files, API endpoints, provider credentials, service dashboards, or shell history.
- Raw logs, training histories, checkpoints, `wandb/`, `mlruns/`, `outputs/`, and full prediction dumps.
- Nested external repositories such as benchmark checkouts; document how to obtain them instead.

## Allowed Review Artifacts

- Source code, prompt templates, configs, and small scripts needed to reproduce reported results.
- Small summary files such as `summary.csv`, `summary.json`, and paper figures if they do not include identifying metadata.
- Anonymous artifact manifests with filenames, generation commands, and hashes.
- Redacted public metadata tables. For example, GIF source creator names are removed from checked-in metadata during review.
- Documented third-party or upstream identifiers, when they are clearly marked as provenance rather than submission authorship.

## Pre-Share Audit

Run a text scan before publishing an anonymous mirror:

```bash
rg -n "author|affiliation|gmail|huggingface|github.com/.+/.+|/home/[^ ]+|Users/|Overleaf|wandb|<known-user-or-org-token>" .
```

Investigate every hit. Replace `<known-user-or-org-token>` with private names, usernames, lab names, and dataset account names before running the scan locally. Some third-party URLs in dependency files may be harmless, but project ownership links and personal paths should be removed or replaced with anonymous placeholders.
