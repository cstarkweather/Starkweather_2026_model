function info = prescreen_anchor_start(x, offers, simcfg)

    params = unpack_dyn(x);

    t_appr  = simcfg.anchor_appr_type;
    t_avoid = simcfg.anchor_avoid_type;

    % ----- fixed-point structure at anchor offers -----
    [fps_appr, nStable_appr]   = count_fixed_points_offer(params, ...
        offers.reward(t_appr), offers.punish(t_appr));

    [fps_avoid, nStable_avoid] = count_fixed_points_offer(params, ...
        offers.reward(t_avoid), offers.punish(t_avoid));

    % stable-point directional biases
    if isempty(fps_appr)
        best_appr_bias = -inf;
    else
        best_appr_bias = max(fps_appr(:,1) - fps_appr(:,2));   % want large positive
    end

    if isempty(fps_avoid)
        best_avoid_bias = inf;
    else
        best_avoid_bias = min(fps_avoid(:,1) - fps_avoid(:,2)); % want large negative
    end

    % ----- neutral-start trajectory sanity checks (optional) -----
    [trajx_appr, trajy_appr] = simulate_from_init( ...
        offers.reward(t_appr), offers.punish(t_appr), params, simcfg, 0, 0);

    [trajx_avoid, trajy_avoid] = simulate_from_init( ...
        offers.reward(t_avoid), offers.punish(t_avoid), params, simcfg, 0, 0);

    b = simcfg.burnin;
    if b > 0 && numel(trajx_appr) > b
        trajx_appr  = trajx_appr(b+1:end);
        trajy_appr  = trajy_appr(b+1:end);
        trajx_avoid = trajx_avoid(b+1:end);
        trajy_avoid = trajy_avoid(b+1:end);
    end

    traj_bias_appr  = mean(trajx_appr - trajy_appr, 'omitnan');
    traj_bias_avoid = mean(trajx_avoid - trajy_avoid, 'omitnan');

    anchor_activity = mean(abs([trajx_appr(:); trajy_appr(:); ...
                                trajx_avoid(:); trajy_avoid(:)]), 'omitnan');

    % ----- noise sanity check -----
    drive_vec = params.lambda * abs(params.alphareward * offers.reward ...
                                  - params.alphapunish * offers.punish);
    drive_max = max(drive_vec);

    if drive_max <= 0
        noise_ratio = inf;
    else
        noise_ratio = params.noise / drive_max;
    end

    % ----- output -----
    info.nStable_appr   = nStable_appr;
    info.nStable_avoid  = nStable_avoid;
    info.best_appr_bias = best_appr_bias;
    info.best_avoid_bias = best_avoid_bias;

    info.traj_bias_appr  = traj_bias_appr;
    info.traj_bias_avoid = traj_bias_avoid;

    info.anchor_activity = anchor_activity;
    info.noise_ratio     = noise_ratio;
end