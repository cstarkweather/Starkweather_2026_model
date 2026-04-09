function [xx, yy, tt] = mfsim_fitting_plot_nullclines_fast( ...
    f1, f2, pausetime, doplot, aReward, aPunish, noise, ...
    lambda, w, a, b, c, visualize_trajectory)

    if doplot || ~visualize_trajectory
        [xx, yy, tt] = mfsim_fitting_plot_nullclines_figure( ...
            f1, f2, pausetime, doplot, aReward, aPunish, noise, ...
            lambda, w, a, b, c, visualize_trajectory);
        return
    end

    tau = 40;
    T   = 2500;

    ExVal = aReward*f1 - aPunish*f2;
    EyVal = 0;

    xx = zeros(1, T);
    yy = zeros(1, T);

    nx = noise * randn(1, T);
    ny = noise * randn(1, T);

    % build params struct for fixed-point finder
    params = struct();
    params.alphareward = aReward;
    params.alphapunish = aPunish;
    params.lambda      = lambda;
    params.w_i         = w;
    params.a           = a;
    params.b           = b;
    params.c           = c;

    % ------------------------------------------------------------
    % Initialize in avoid-like stable fixed point of BASELINE system
    % ------------------------------------------------------------
    [fps0_stable, nStable0, fps0_all, ~] = count_fixed_points_offer(params, 0, 0);

    if isempty(fps0_stable)
        if isempty(fps0_all)
            x = 0;
            y = max(c);
        else
            [~, ix] = max(fps0_all(:,2) - fps0_all(:,1));  % most avoid-like
            x = fps0_all(ix,1);
            y = fps0_all(ix,2);
        end
    else
        [~, ix] = max(fps0_stable(:,2) - fps0_stable(:,1));  % most avoid-like stable
        x = fps0_stable(ix,1);
        y = fps0_stable(ix,2);
    end

    for k = 1:T
        sx = -w*y + lambda*ExVal;
        sy = -w*x + lambda*EyVal;

        fx = a * tanh(sx + b) + c;
        fy = a * tanh(sy + b) + c;

        x = x + (-x + fx + nx(k)) / tau;
        y = y + (-y + fy + ny(k)) / tau;

        xx(k) = w * x;
        yy(k) = w * y;
    end

    tt = 1:T;
end