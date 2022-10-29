---
fed-number: 0001
title: Flower Enhancement Doc
authors: ["@nfnt", "@orlandohohmeier"]
creation-data: 2022-03-11
last-updated: 2022-10-24
status: provisional
---

# Flower Enhancement Doc

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Summary](#summary)
- [Motivation](#motivation)
  - [Goals](#goals)
  - [Non-Goals](#non-goals)
- [Proposal](#proposal)
  - [Enhancement Doc Template](#enhancement-doc-template)
  - [Metadata](#metadata)
  - [Workflow](#workflow)
- [Drawbacks](#drawbacks)
- [Alternatives Considered](#alternatives-considered)
  - [GitHub Issues](#github-issues)
  - [Google Docs](#google-docs)

## Summary

A Flower Enhancement is a standardized development process to

- provide a common structure for proposing larger changes
- ensure that the motivation for a change is clear
- persist project information in a version control system
- document the motivation for impactful user-facing changes
- reserve GitHub issues for tracking work in flight
- ensure community participants can successfully drive changes to completion across one or more releases while stakeholders are adequately represented throughout the process

Hence, an Enhancement Doc combines aspects of

- a feature, and effort-tracking document
- a product requirements document
- a design document

into one file, which is created incrementally in collaboration with the community.

## Motivation

For far-fetching changes or features proposed to Flower, an abstraction beyond a single GitHub issue or pull request is required to understand and communicate upcoming changes to the project.

The purpose of this process is to reduce the amount of "tribal knowledge" in our community. By moving decisions from Slack threads, video calls, and hallway conversations into a well-tracked artifact, this process aims to enhance communication and discoverability.

### Goals

Roughly any larger, user-facing enhancement should follow the Enhancement process. If an enhancement would be described in either written or verbal communication to anyone besides the author or developer, then consider creating an Enhancement Doc.

Similarly, any technical effort (refactoring, major architectural change) that will impact a large section of the development community should also be communicated widely. The Enhancement process is suited for this even if it will have zero impact on the typical user or operator.

### Non-Goals

For small changes and additions, going through the Enhancement process would be time-consuming and unnecessary. This includes, for example, adding new Federated Learning algorithms, as these only add features without changing how Flower works or is used.

Enhancements are different from feature requests, as they are already providing a laid-out path for implementation and are championed by members of the community.

## Proposal

An Enhancement is captured in a Markdown file that follows a defined template and a workflow to review and store enhancement docs for reference — the Enhancement Doc.

### Enhancement Doc Template

Each enhancement doc is provided as a Markdown file having the following structure

- Metadata (as [described below](#metadata) in form of a YAML preamble)
- Title (same as in metadata)
- Table of Contents (if needed)
- Summary
- Motivation
  - Goals
  - Non-Goals
- Proposal
  - Notes/Constraints/Caveats (optional)
- Design Details (optional)
  - Graduation Criteria
  - Upgrade/Downgrade Strategy (if applicable)
- Drawbacks
- Alternatives Considered

As a reference, this document follows the above structure.

### Metadata

- **fed-number** (Required)
  The `fed-number` of the last Flower Enhancement Doc + 1. With this number, it becomes easy to reference other proposals.
- **title** (Required)
  The title of the proposal in plain language.
- **status** (Required)
  The current status of the proposal. See [workflow](#workflow) for the possible states.
- **authors** (Required)
  A list of authors of the proposal. This is simply the GitHub ID.
- **creation-date** (Required)
  The date that the proposal was first submitted in a PR.
- **last-updated** (Optional)
  The date that the proposal was last changed significantly.
- **see-also** (Optional)
  A list of other proposals that are relevant to this one.
- **replaces** (Optional)
  A list of proposals that this one replaces.
- **superseded-by** (Optional)
  A list of proposals that this one supersedes.

### Workflow

The idea forming the enhancement should already have been discussed or pitched in the community. As such, it needs a champion, usually the author, who shepherds the enhancement. This person also has to find committers to Flower willing to review the proposal.

New enhancements are checked in with a file name in the form of `NNNN-YYYYMMDD-enhancement-title.md`, with `NNNN` being the Flower Enhancement Doc number, to `enhancements`. All enhancements start in `provisional` state as part of a pull request. Discussions are done as part of the pull request review.

Once an enhancement has been reviewed and approved, its status is changed to `implementable`. The actual implementation is then done in separate pull requests. These pull requests should mention the respective enhancement as part of their description. After the implementation is done, the proposal status is changed to `implemented`.

Under certain conditions, other states are possible. An Enhancement has the following states:

- `provisional`: The enhancement has been proposed and is actively being defined. This is the starting state while the proposal is being fleshed out and actively defined and discussed.
- `implementable`: The enhancement has been reviewed and approved.
- `implemented`: The enhancement has been implemented and is no longer actively changed.
- `deferred`: The enhancement is proposed but not actively being worked on.
- `rejected`: The authors and reviewers have decided that this enhancement is not moving forward.
- `withdrawn`: The authors have withdrawn the enhancement.
- `replaced`: The enhancement has been replaced by a new enhancement.

## Drawbacks

Adding an additional process to the ones already provided by GitHub (Issues and Pull Requests) adds more complexity and can be a barrier for potential first-time contributors.

Expanding the proposal template beyond the single-sentence description currently required in the features issue template may be a heavy burden for non-native English speakers.

## Alternatives Considered

### GitHub Issues

Using GitHub Issues for these kinds of enhancements is doable. One could use, for example, tags, to differentiate and filter them from other issues. The main issue is in discussing and reviewing an enhancement: GitHub issues only have a single thread for comments. Enhancements usually have multiple threads of discussion at the same time for various parts of the doc. Managing these multiple discussions can be confusing when using GitHub Issues.

### Google Docs

Google Docs allow for multiple threads of discussions. But as Google Docs are hosted outside the project, their discoverability by the community needs to be taken care of. A list of links to all proposals has to be managed and made available for the community. Compared to shipping proposals as part of Flower's repository, the potential for missing links is much higher.
