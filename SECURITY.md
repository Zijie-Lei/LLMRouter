# Security

## Report a vulnerability

Use [GitHub's private vulnerability reporting](https://github.com/ulab-uiuc/LLMRouter/security/advisories/new)
when it is enabled. If that option is unavailable, contact an author listed in
`pyproject.toml` privately. Do not post credentials or sensitive exploit details
in a public issue.

Include the affected version, installation/deployment details, a minimal
reproduction, and the impact. Remove API keys and user data from reports.

## Versions and deployment

Security fixes are developed against `main` and published in a new release.
There is currently no commitment to backport fixes to older versions.

Treat API keys, router configuration, model artifacts, and plugins as trusted
inputs. Deploy HTTP services behind authentication and request limits suited to
your environment. Review and pin dependencies for deployed applications.

PR dependency review blocks newly introduced known high/critical advisories.
Dependabot proposes updates and CodeQL analyzes Python code. These checks do not
establish that a deployment or all existing dependencies are secure.
