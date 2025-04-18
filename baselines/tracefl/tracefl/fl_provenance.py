import logging


from tracefl.fl_provenance_modules import FederatedProvTrue


# --------------------------------------------------------------------
# round_lambda_prov
# --------------------------------------------------------------------
def round_lambda_prov(
    train_cfg, 
    prov_cfg, 
    round_key,
    central_test_data,
    client2model,
    client2num_examples,
    prov_global_model,
    ALLROUNDSCLIENTS2CLASS,
):
    # If the train_cfg indicates we have faulty clients

    round_prov = FederatedProvTrue(
        train_cfg,
        prov_cfg,
        round_key,
        central_test_data,
        client2model,
        client2num_examples,
        prov_global_model,
        ALLROUNDSCLIENTS2CLASS,
        t=None,
    )

    try:
        prov_result_dict = round_prov.run()
    except Exception as e:
        logging.error(f"Error in running provenance for round {round_key}. Error: {e}")
        prov_result_dict = {"Error": str(e)}

    return prov_result_dict
