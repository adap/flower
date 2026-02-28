# Always-Auth SuperGrid

## Motivation

Today, the SuperLink can run in two or three configurations, depending on how you count:
- Auth enabled [available only on SuperGrid]
  - (Option 1) Auth enabled using Flower Account
  - (Option 2) Auth enabled using 3rd party OIDC server
- (Option 3) Auth disabled [available on both SuperGrid and Framework]

There are a few downsides with the current approach:
- This requires us to think about every feature twice: how does it work with or without auth?
- It may also be unclear to open-source users that Flower actually supports user accounts at all. 
- We need to maintain and test both auth and no-auth.
- With upcoming agent support, we need user accounts to let the user call 
- 

With upcoming agent support in SuperLink, it'd be useful to have Flower Account support in Framework. Flower Account would enable Framework users to log in and use their account to retrieve use featues of their Flower Account subsription via the open-source Framework. For example, the SuperLink could use the account to get an `API_KEY` for the Flower Intelligence API to enable locally running agents (on the open-source SuperLink) to call the Flower Intelligence API. Other features could include logging local runs to SuperGrid (if we want to do this, which is TBD).

### Goals

- Enable Framework users to use Flower Account: 
- More consistent UX between Framework and SuperGrid: users should have a consistent user experience across Framework and SuperGrid.
- Educate open-source users more of the power of Flower: 
- Simplify our architecture: remove support for the no-auth configuration.
- Get more users to use Flower Account: 
- Keep first-user experience simple: 

### Non-goal

- Make Flower Account the *only* auth solution

## Proposal

This RFC proposes to remove *Option 1: Auth disabled*. This would change the user experience in a way where it always asks the user to log in. It would leave the user with two options:

- Option 1 [**new default**]: Auth enabled using Flower Account
- Option 2: Auth enabled using 3rd party OIDC server

Having the 

### Product rationale

Removing no-auth would make the UX of SuperGrid and Framework users more consistent and reduce the technical complexity. Users would always log-in to SuperLink, either using the Flower Account (default) or a custom OIDC server. Having the Flower Account in open-source will provide an 

Upside:
- More consistent UX between Framework and SuperGrid: All users would use Flower with auth enabled.
- User education that Flower supports multi-account usage: Today, Framework users might not even be aware of the fact that Flower supports user accounts.
- Reduce technical complexity: we could remove a bunch of code and testing scenarios b/c every user would use Flower with auth enabled.
- Flower Account gets used more widely: This makes the Flower Account the new default for Federated AI.
- Improved telemetry: If the Flower Account becomes the new default for open-source, it will give us more insights into how open-source users use Flower.

Downsides:
- First-time users have to log in: this might not be a big issue b/c we want to update the first-user experience to use SuperGrid anyways. That means the first run a user does when following the tutorial should be on SuperGrid (simulation mode). Running your own SuperLink will be positioned as an option for more advanced users.

Open questions:
- Do we want to allow open-source users to customize the OIDC server? Or only expose that option for SuperGrid users (which means open-source users would always use Flower Account)?

### Technical implications

Removing the no-auth option would simplify our code, but also require us to fully open-source OIDC support in the SuperLink.

### Open KeyCloak question: `client_id` and `client_secret`?

With our current auth approach, KeyCloak requires a `client_id` and a `client_secret` to be configured.

If we support Flower Account in the open-source Framework, we cannot hardcode the `client_id` and `client_secret` we use for SuperGrid. We could create a new `client_id` and `client_secret`. But there might also be alternative auth flows that are more appropriate for the intended setup.

The intended setup turns the open-source SuperLink more into an untrusted KeyCloak client (as opposed to a trusted one that runs in our infrastucture). There may be a different OIDC flow that better fits this setup.

### Action items

- Review & decide KeyCloak auth flow question
- Open-source full OIDC implementation
  - Remaining open question: make OIDC server configurable?
- Remove the `--disable-auth` option
- Clean up code: remove all remaining traces of no-auth
