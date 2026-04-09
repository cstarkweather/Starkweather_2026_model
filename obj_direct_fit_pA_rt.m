function [err, pred] = obj_direct_fit_pA_rt(x, target, offers, simcfg)

    % default large penalty
    err  = 1e6;
    pred = struct();

    try
        params = unpack_dyn(x);
        post   = get_anchor_cache(x, offers, simcfg);
        pred   = simulate_model_summaries_cached(params, offers, simcfg, post);

        % -------------------- modeled summaries --------------------
        pA_model   = pred.pA(:);
        rt_model   = pred.rt(:);
        conf_model = pred.conf(:);

        % -------------------- target summaries --------------------
        pA_target   = target.pA_mean(:);
        rt_target   = target.rt_mean(:);
        conf_target = target.conf_mean(:);

        % -------------------- reject invalid model output --------------------
        if any(~isfinite(pA_model)) || any(~isfinite(rt_model)) || any(~isfinite(conf_model))
            pred.err_pA    = nan;
            pred.err_rt    = nan;
            pred.err_conf  = nan;
            pred.err_total = err;
            return
        end

        % -------------------- (1) pA fit --------------------
        dpA = pA_model - pA_target;
        ok  = isfinite(dpA);

        if any(ok)
            err_pA = mean(dpA(ok).^2);
        else
            err_pA = 1e6;
        end

        % -------------------- (2) RT fit: z-scored across offers --------------------
        rt_model_z  = nan(size(rt_model));
        rt_target_z = nan(size(rt_target));
        drt         = nan(size(rt_model));

        % remove degenerate fits with zero RT variance across offers
        if sum(isfinite(rt_model)) < 2 || std(rt_model,'omitnan') <= 0
            err_rt = 1e6;
        elseif sum(isfinite(rt_target)) < 2 || std(rt_target,'omitnan') <= 0
            err_rt = 1e6;
        else
            rt_model_z  = (rt_model  - mean(rt_model,'omitnan'))  ./ std(rt_model,'omitnan');
            rt_target_z = (rt_target - mean(rt_target,'omitnan')) ./ std(rt_target,'omitnan');

            drt = rt_model_z - rt_target_z;
            ok  = isfinite(drt);

            if any(ok)
                err_rt = mean(drt(ok).^2);
            else
                err_rt = 1e6;
            end
        end

        % -------------------- (3) conflict fit --------------------
        dconf = conf_model - conf_target;
        ok    = isfinite(dconf);

        if any(ok)
            err_conf = mean(dconf(ok).^2);
        else
            err_conf = 1e6;
        end

        % -------------------- equally weighted objective --------------------
        err = err_pA + err_rt + err_conf;

        if ~isfinite(err) || ~isreal(err)
            err = 1e6;
        end

        % -------------------- diagnostics --------------------
        pred.err_pA    = err_pA;
        pred.err_rt    = err_rt;
        pred.err_conf  = err_conf;
        pred.err_total = err;

        pred.rt_model_z  = rt_model_z;
        pred.rt_target_z = rt_target_z;
        pred.dpA         = dpA;
        pred.drt         = drt;
        pred.dconf       = dconf;

    catch
        err = 1e6;
        pred.err_pA    = nan;
        pred.err_rt    = nan;
        pred.err_conf  = nan;
        pred.err_total = err;
    end
end