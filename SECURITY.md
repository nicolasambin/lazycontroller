# Security policy

## Supported versions

lazycontroller is in alpha. Security fixes go into the latest 0.x release only; upgrade to get
them.

## Reporting a vulnerability

Please **don't open a public issue**. Report it privately through GitHub: on the repository's
**Security** tab, choose **Report a vulnerability**. Include what you found, how to reproduce
it, and which version you used.

This is a small, maintainer-run project: reports are handled on a best-effort basis. You'll get
an acknowledgement, and once a fix is released the advisory is published with credit to you,
unless you'd rather stay anonymous.

## What's in scope

lazycontroller sits between your code and two model providers and handles their API keys, so
the issues that matter most are about where data and keys go:

- an API key sent to a server other than the one it belongs to (for example `FastConfig.api_key`
  reaching anything but `FastConfig.base_url`, or the slow side's key reaching the fast model's
  server);
- conversation content sent to the fast model when it shouldn't be: with `enabled=False`,
  without `base_url`, or for a request lazycontroller should have left alone;
- a request lazycontroller should pass through untouched (raw responses, streaming, extra
  request options) being altered or answered by the fast model;
- anything that lets input in a conversation change which server lazycontroller talks to or
  which credentials it uses.

## Out of scope

- Vulnerabilities in the `openai` or `typesafe-sdk` packages, or in a model provider's service:
  report those to their maintainers. Known vulnerabilities in dependencies are tracked here
  with Dependabot.
- The quality of a model's answers, including answers swayed by text in the conversation: the
  fast model can only pick among the values your schema allows, and `min_confidence`
  decides when its answer is used.
