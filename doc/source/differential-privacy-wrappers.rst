Differential Privacy Wrappers in Flower
=======================================

Flower provides differential privacy (DP) wrapper classes for the easy integration of the central DP guarantees provided by DP-FedAvg into training pipelines defined in any of the various ML frameworks that Flower is compatible with. 

.. note::
  The name DP-FedAvg is misleading since it can be applied on top of any FL algorithm that conforms to the general structure prescribed by the FedOpt family of algorithms.

DP-FedAvg
---------

DP-FedAvg, originally proposed by McMahan et al. [mcmahan]_ and extended by Andrew et al. [andrew]_, is essentially FedAvg with the following modifications. 

* **Clipping** : The influence of each client's update is bounded by clipping it. This is achieved by enforcing a cap on the L2 norm of the update, scaling it down if needed.
* **Noising** :  Gaussian noise, calibrated to the clipping threshold, is added to the average computed at the server.

The distribution of the update norm has been shown to vary from task-to-task and to evolve as training progresses. Therefore, we use an adaptive approach [andrew]_ that continuously adjusts the clipping threshold to track a prespecified quantile of the update norm distribution. 

Simplifying Assumptions
***********************

We make (and attempt to enforce) a number of assumptions that must be satisfied to ensure that the training process actually realises the :math:`(\epsilon, \delta)` guarantees the user has in mind when configuring the setup. 

* **Fixed-size subsampling** :Fixed-size subsamples of the clients must be taken at each round, as opposed to variable-sized Poisson subsamples. 
* **Unweighted averaging** : The contributions from all the clients must weighted equally in the aggregate to eliminate the requirement for the server to know in advance the sum of the weights of all clients available for selection.
* **No client failures** : The set of available clients must stay constant across all rounds of training. In other words, clients cannot drop out or fail. 

The first two are useful for eliminating a multitude of complications associated with calibrating the noise to the clipping threshold while the third one is required to comply with the assumptions of the privacy analysis.

.. note::
   These restrictions are in line with constraints imposed by Andrew et al. [andrew]_.

Customizable Responsibility for Noise injection
***********************************************
In contrast to other implementations where the addition of noise is performed at the server, you can configure the site of noise injection to better match your threat model. We provide users with the flexibility to set up the training such that each client independently adds a small amount of noise to the clipped update, with the result that simply aggregating the noisy updates is equivalent to the explicit addition of noise to the non-noisy aggregate at the server. 


To be precise, if we let :math:`m` be the number of clients sampled each round and :math:`\sigma_\Delta` be the scale of the total Gaussian noise that needs to be added to the sum of the model updates, we can use simple maths to show that this is equivalent to each client adding noise with scale :math:`\sigma_\Delta/\sqrt{m}`.

Wrapper-based approach
----------------------

Introducing DP to an existing workload can be thought of as adding an extra layer of security around it. This inspired us to provide the additional server and client-side logic needed to make the training process differentially private as wrappers for instances of the :code:`Strategy` and :code:`NumPyClient` abstract classes respectively. This wrapper-based approach has the advantage of being easily composable with other wrappers that someone might contribute to the Flower library in the future, e.g., for secure aggregation. Using Inheritance instead can be tedious because that would require the creation of new sub- classes every time a new class implementing :code:`Strategy` or :code:`NumPyClient` is defined.

Server-side logic
*****************

The first version of our solution was to define a decorator whose constructor accepted, among other things, a boolean valued variable indicating whether adaptive clipping was to be enabled or not. We quickly realized that this would clutter its :code:`__init__()` function with variables corresponding to hyperparameters of adaptive clipping that would remain unused when it was disabled. A cleaner implementation could be achieved by splitting the functionality into two decorators, :code:`DPFedAvgFixed` and :code:`DPFedAvgAdaptive`, with the latter sub- classing the former. The constructors for both classes accept a boolean parameter :code:`server_side_noising`, which, as the name suggests, determines where noising is to be performed.

DPFedAvgFixed
:::::::::::::

The server-side capabilities required for the original version of DP-FedAvg, i.e., the one which performed fixed clipping, can be completely captured with the help of wrapper logic for just the following two methods of the :code:`Strategy` abstract class.

#. :code:`configure_fit()` : The config dictionary being sent by the wrapped :code:`Strategy` to each client needs to be augmented with an additional value equal to the clipping threshold (keyed under :code:`dpfedavg_clip_norm`) and, if :code:`server_side_noising=true`, another one equal to the scale of the Gaussian noise that needs to be added at the client (keyed under :code:`dpfedavg_noise_stddev`). This entails *post*-processing of the results returned by the wrappee's implementation of :code:`configure_fit()`.
#. :code:`aggregate_fit()`: We check whether any of the sampled clients dropped out or failed to upload an update before the round timed out. In that case, we need to abort the current round, discarding any successful updates that were received, and move on to the next one. On the other hand, if all clients responded successfully, we must force the averaging of the updates to happen in an unweighted manner by intercepting the :code:`parameters` field of :code:`FitRes` for each received update and setting it to 1. Furthermore, if :code:`server_side_noising=true`, each update is perturbed with an amount of noise equal to what it would have been subjected to had client-side noising being enabled.  This entails *pre*-processing of the arguments to this method before passing them on to the wrappee's implementation of :code:`aggregate_fit()`.

.. note::
  We can't directly change the aggregation function of the wrapped strategy to force it to add noise to the aggregate, hence we simulate client-side noising to implement server-side noising. 

These changes have been put together into a class called :code:`DPFedAvgFixed`, whose constructor accepts the strategy being decorated, the clipping threshold and the number of clients sampled every round as compulsory arguments. The user is expected to specify the clipping threshold since the order of magnitude of the update norms is highly dependent on the model being trained and providing a default value would be misleading. The number of clients sampled at every round is required to calculate the amount of noise that must be added to each individual update, either by the server or the clients. 

DPFedAvgAdaptive
::::::::::::::::

The additional functionality required to facilitate adaptive clipping has been provided in :code:`DPFedAvgAdaptive`, a subclass of :code:`DPFedAvgFixed`. It overrides the above-mentioned methods to do the following. 

#. :code:`configure_fit()` : It intercepts the config dict returned by :code:`super.configure_fit()` to add the key-value pair :code:`dpfedavg_adaptive_clip_enabled:True`to it, which the client interprets as an instruction to include an indicator bit (1 if update norm <= clipping threshold, 0 otherwise) in the results returned by it. 

#. :code:`aggregate_fit()` : It follows a call to :code:`super.aggregate_fit()` with one to :code:`__update_clip_norm__()`, a procedure which adjusts the clipping threshold on the basis of the indicator bits received from the sampled clients. 


Client-side logic
*****************

The client-side capabilities required can be completely captured through wrapper logic for just the :code:`fit()` method of the :code:`NumPyClient` abstract class. To be precise, we need to *post-process* the update computed by the wrapped client to clip it, if necessary, to the threshold value supplied by the server as part of the config dictionary. In addition to this, it may need to perform some extra work if either (or both) of the following keys are also present in the dict.

* :code:`dpfedavg_noise_stddev` : Generate and add the specified amount of noise to the clipped update.
* :code:`dpfedavg_adaptive_clip_enabled` : Augment the metrics dict in the :code:`FitRes` object being returned to the server with an indicator bit, calculated as described earlier.


Performing the :math:`(\epsilon, \delta)` analysis
--------------------------------------------------

Assume you have trained for :math:`n` rounds with sampling fraction :math:`q` and noise multiplier :math:`z`. In order to calculate the :math:`\epsilon` value this would result in for a particular :math:`\delta`, the following script may be used. 

.. code-block:: python
   import tensorflow_privacy as tfp
   max_order = 32
   orders = range(2, max_order + 1)
   rdp = tfp.compute_rdp_sample_without_replacement(q, z, n, orders)
   eps, _, _ = tfp.rdp_accountant.get_privacy_spent(rdp, target_delta=delta)

.. [mcmahan] McMahan, H. Brendan, et al. "Learning differentially private recurrent language models." arXiv preprint arXiv:1710.06963 (2017).

.. [andrew] Andrew, Galen, et al. "Differentially private learning with adaptive clipping." Advances in Neural Information Processing Systems 34 (2021): 17455-17466.
