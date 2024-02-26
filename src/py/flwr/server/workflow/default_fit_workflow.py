


def default_fit_workflow(driver: Driver, context: Context) -> None:
    """Execute the default workflow for a single fit round."""
    if not isinstance(context, LegacyContext):
        raise TypeError(f"Expect a LegacyContext, but get {type(context).__name__}.")

    # Get current_round and parameters
    current_round = cast(
        int, context.state.configs_records[CONFIGS_RECORD_KEY][KEY_CURRENT_ROUND]
    )
    parametersrecord = context.state.parameters_records[PARAMS_RECORD_KEY]
    parameters = compat.parametersrecord_to_parameters(
        parametersrecord, keep_input=True
    )

    # Get clients and their respective instructions from strategy
    client_instructions = context.strategy.configure_fit(
        server_round=current_round,
        parameters=parameters,
        client_manager=context.client_manager,
    )

    if not client_instructions:
        log(INFO, "fit_round %s: no clients selected, cancel", current_round)
        return
    log(
        DEBUG,
        "fit_round %s: strategy sampled %s clients (out of %s)",
        current_round,
        len(client_instructions),
        context.client_manager.num_available(),
    )

    # Build dictionary mapping node_id to ClientProxy
    node_id_to_proxy = {proxy.node_id: proxy for proxy, _ in client_instructions}

    # Build out messages
    out_messages = [
        driver.create_message(
            content=compat.fitins_to_recordset(fitins, True),
            message_type=MESSAGE_TYPE_FIT,
            dst_node_id=proxy.node_id,
            group_id="",
            ttl="",
        )
        for proxy, fitins in client_instructions
    ]

    # Send instructions to clients and
    # collect `fit` results from all clients participating in this round
    messages = list(driver.send_and_receive(out_messages))
    del out_messages

    # No exception/failure handling currently
    log(
        DEBUG,
        "fit_round %s received %s results and %s failures",
        current_round,
        len(messages),
        0,
    )

    # Aggregate training results
    results = [
        (
            node_id_to_proxy[msg.metadata.src_node_id],
            compat.recordset_to_fitres(msg.content, False),
        )
        for msg in messages
    ]
    aggregated_result = context.strategy.aggregate_fit(current_round, results, [])
    parameters_aggregated, metrics_aggregated = aggregated_result

    # Update the parameters and write history
    if parameters_aggregated:
        paramsrecord = compat.parameters_to_parametersrecord(
            parameters_aggregated, True
        )
        context.state.parameters_records[PARAMS_RECORD_KEY] = paramsrecord

    context.history.add_metrics_distributed_fit(
        server_round=current_round, metrics=metrics_aggregated
    )
    
