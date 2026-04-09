function pred = simulate_model_summaries_cached(params, offers, simcfg, post)

    rng(simcfg.rngSeed);

    nTypes = numel(offers.reward);
    nT = simcfg.n_trials_sim;

    pA = nan(1,nTypes);
    rt = nan(1,nTypes);

    for t = 1:nTypes
        isApproach = nan(nT,1);
        dts        = nan(nT,1);

        for k = 1:nT
            [trajx,trajy] = mfsim_fitting_plot_nullclines_fast( ...
                offers.reward(t), offers.punish(t), ...
                0, 0, ...
                params.alphareward, params.alphapunish, params.noise, ...
                params.lambda, params.w_i, params.a, ...
                params.b, params.c, 1);

            b0 = simcfg.burnin;
            if b0 > 0 && numel(trajx) > b0
                trajx = trajx(b0+1:end);
                trajy = trajy(b0+1:end);
            end

            p = compute_posterior_timeseries(trajx, trajy, post);

            if simcfg.smooth_post > 1
                p = smooth(p, simcfg.smooth_post);
            end

            [dt, dec] = detectStatePreferences(p, params.time_stable, params.thresh);

            if ~isnan(dec)
                isApproach(k) = double(dec == 1);
            end
            dts(k) = dt;
        end

        pA(t) = mean(isApproach, 'omitnan');
        rt(t) = mean(dts, 'omitnan');
    end

    % entropy-based conflict from offer-wise approach probability
    pA_clip = min(max(pA, 1e-6), 1 - 1e-6);
    pAvoid  = 1 - pA_clip;

    conf = -(pA_clip .* log2(pA_clip) + pAvoid .* log2(pAvoid));

    pred.pA   = pA;
    pred.rt   = rt;
    pred.conf = conf;
end