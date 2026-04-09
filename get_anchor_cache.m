function post = get_anchor_cache(x, offers, simcfg)

    persistent last_x last_post

    if ~isempty(last_x) && isequal(x, last_x)
        post = last_post;
        return
    end

    params = unpack_dyn(x);
    rng(simcfg.rngSeed);

    rA = offers.reward(simcfg.anchor_appr_type);
    pA = offers.punish(simcfg.anchor_appr_type);

    rV = offers.reward(simcfg.anchor_avoid_type);
    pV = offers.punish(simcfg.anchor_avoid_type);

    nRep = simcfg.anchor_trials;

    XA = [];
    YA = [];
    XV = [];
    YV = [];

    for k = 1:nRep
        [xA, yA] = mfsim_fitting_plot_nullclines_fast( ...
            rA, pA, 0, 0, ...
            params.alphareward, params.alphapunish, params.noise, ...
            params.lambda, params.w_i, params.a, params.b, params.c, 1);

        [xV, yV] = mfsim_fitting_plot_nullclines_fast( ...
            rV, pV, 0, 0, ...
            params.alphareward, params.alphapunish, params.noise, ...
            params.lambda, params.w_i, params.a, params.b, params.c, 1);

        burninN = simcfg.burnin;

        if burninN > 0 && numel(xA) > burninN
            xA = xA(burninN+1:end);
            yA = yA(burninN+1:end);
        end

        if burninN > 0 && numel(xV) > burninN
            xV = xV(burninN+1:end);
            yV = yV(burninN+1:end);
        end

        tailNA = min(500, numel(xA));
        tailNV = min(500, numel(xV));

        XA = [XA; xA(end-tailNA+1:end).'];
        YA = [YA; yA(end-tailNA+1:end).'];
        XV = [XV; xV(end-tailNV+1:end).'];
        YV = [YV; yV(end-tailNV+1:end).'];
    end

    % store raw samples too
    post.XA = XA(:);
    post.YA = YA(:);
    post.XV = XV(:);
    post.YV = YV(:);

    post.mu_x_appr = mean(XA);
    post.sd_x_appr = std(XA) + eps;
    post.mu_y_appr = mean(YA);
    post.sd_y_appr = std(YA) + eps;

    post.mu_x_av   = mean(XV);
    post.sd_x_av   = std(XV) + eps;
    post.mu_y_av   = mean(YV);
    post.sd_y_av   = std(YV) + eps;

    post.pi_appr = 0.5;
    post.pi_av   = 0.5;

    last_x    = x;
    last_post = post;
end