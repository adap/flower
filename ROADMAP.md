# Roadmap

## Big topics

Ongoing:
- Federation Management

Later>
- SuperGrid UI -> Stephane will work on it (up to 4 weeks, mid-Feb - mid-Mar)
- Checkout -> Stephane will work on it (up to 4 weeks, mid-Mar - mid-Apr)
- Confidential compute: Patrick and Mohammad will aim to have a "Preview Support" at FAIS26
- Dependency management -> no plan yet
- Better roles/permissions


- Reliability:
  - Postgres support (Chong Shen)
  - Versioning between all components
    - Internal version checks:
        - CLI <> SuperLink
        - SuperNode <> SuperLink
        - flwr-serverapp <> SuperLink
        - flwr-clientapp <> SuperNode
    - Update checks:
        - CLI / SuperLink / SuperNode / flwr-*app <> platform-api (is a new version available?)
    - Other:
        - Hub: each app needs to declare 
        - CLI: Flower Hub needs to resolve version compatibility

- Simplification

## Radical Simplification

- Move from GitLab to GitHub
- 60s CI (10m -> 5m -> 60s)
- Adopt `uv` for every Python project
- Migrate
- Align every Python project: `dev` dir with `format.sh` / `build.sh` / `test.sh`

System architecture:

Today:
- telemetry.flower.ai -> `telemetry-api` (public telemetry endpoint)
- metrics.flower.ai -> `metrics-api`
- account.flower.ai -> OIDC server (keycloak)
- api.flower.ai -> `platform-api` (REST)
- supergrid.flower.ai -> SuperLink `Control API` (gRPC)
- fleet-supergrid.flower.ai -> SuperLink `Control API` (gRPC)
- Internal:
  - SuperLink ServerAppIo API
  - 


Next:
- telemetry.flower.ai -> public telemetry endpoint
- account.flower.ai -> OIDC server
- api.flower.ai / supergrid.flower.ai -> 
- Internal:
  - SuperExec -> 

How?
- Merge `metrics-api` into `telemetry-api`
- 

DevOps:







- data: compare central data vs more distributed data + DP
- compute: wishful thinking, would decent lower compute cost?
- pretraining: just a marginal part of the whole training process?
- pitch: undersold ourselves
- people were starting to look at operational + technical details b/c we failed to show the big vision
- how do we use distribution to unlock new frontier at scale
- interested in operational detail, failed to see what the full potential could be
- try to apply to the big challenge: give people the impression of what can it be? why do we need that? what do we unlock? where can it lead to? what's our angle? make people dream
- apply to the big challenge: yes
- we weren't qualitativley worse than the other teams
  - we've proven that we have technical excellence, operational excellence -- can we really pivot into the frontier lab track? can we convincingly argue that we pivot? can re realign the company?
- pitch that came afterwards for CL challenge: was much better thant the frontier ai pitch. played to our strengths, 
- FL: underdelivered, not next frontier
- other part: frontier is data driven, in federated youre not the data broker, how do you get the data? frontier system that does not require TBs of data
- other frontiers: more verticalized full-stack systems, people have a clear picture of what this could be
- tell people a really big dream 
- one variant of this dream: if we go into this direction of unlocking industrial data, project we do with SAP, tease people in this direction, unlock industrial data on huge scale, Europe sitting on huge pile of industrial data, people buy into that
- imagine you had a billion of ...., we go on a bying spree and buy data packages all around to unlock it -- why do we need the 27m EUR to prepare for that?
- 

