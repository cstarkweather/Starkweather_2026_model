function [xx, yy, tt] = mfsim_fitting_plot_nullclines_figure( ...
    f1, f2, pausetime, doplot, aReward, aPunish, noise, lambda, w, a, b, c, visualize_trajectory)

    xgrd = -12000:0.1:12000;
    tau  = 40;
    T    = 10;
    t    = 1:T;

    % continuous input for full trial
    Ex = (aReward*f1 - aPunish*f2) * ones(1, T);
    Ey = zeros(1, T);

    pathHandle = [];
    posxy      = [];
    ttHandle   = [];

    if doplot
        pos = get(gcf,'Position');
        set(gcf,'Position', [pos(1) pos(2) 360 600]);

        hold on;
        nc1 = plot(xgrd, xgrd, 'k');
        nc2 = plot(xgrd, xgrd, 'k');

        set(nc1, 'Color', 'k', 'LineWidth', 2);
        set(nc2, 'Color', 'k', 'LineWidth', 2);

        if visualize_trajectory
            pathHandle = plot(NaN, NaN, 'k-', 'LineWidth', 1);
            posxy      = plot(NaN, NaN, 'r.', 'MarkerSize', 20);
            ttHandle   = text(3, 4.5, 't=0');
        end

        xlabel('nS (plus neuron)');
        ylabel('nS (minus neuron)');
        axis([-0.5 12 -0.5 12]);
        set(gca, 'TickDir', 'out');
    end

    % ------------------------------------------------------------
    % Build params struct for fixed-point finder
    % ------------------------------------------------------------
    params = struct();
    params.alphareward = aReward;
    params.alphapunish = aPunish;
    params.lambda      = lambda;
    params.w_i         = w;
    params.a           = a;
    params.b           = b;
    params.c           = c;

    % ------------------------------------------------------------
    % Initialize in avoid-like fixed point of BASELINE system
    % ------------------------------------------------------------
    [fps0, isStable0] = count_fixed_points_offer(params, 0, 0);

    if isempty(fps0)
        x = 0;
        y = max(c);
    else
        fps0_stable = fps0(isStable0 == 1, :);

        if isempty(fps0_stable)
            fps_use = fps0;
        else
            fps_use = fps0_stable;
        end

        [~, ix] = max(fps_use(:,2) - fps_use(:,1));  % most avoid-like
        x = fps_use(ix,1);
        y = fps_use(ix,2);
    end

    xx = zeros(1, T);
    yy = zeros(1, T);

    xPath = [];
    yPath = [];

    for k = 1:T

        sx = -w*y + lambda*Ex(k);
        sy = -w*x + lambda*Ey(k);

        dx = -x + iofunc(sx, a, b, c) + noise*randn;
        dy = -y + iofunc(sy, a, b, c) + noise*randn;

        x = x + dx/tau;
        y = y + dy/tau;

        xx(k) = w*x;
        yy(k) = w*y;

        if doplot
            set(nc1, 'XData', xgrd, ...
                     'YData', w*iofunc(-xgrd + lambda*Ey(k), a, b, c));

            set(nc2, 'XData', w*iofunc(-xgrd + lambda*Ex(k), a, b, c), ...
                     'YData', xgrd);

            if visualize_trajectory
                xPath = [xPath, w*x];
                yPath = [yPath, w*y];

                set(pathHandle, 'XData', xPath, 'YData', yPath);
                set(posxy, 'XData', w*x, 'YData', w*y);
                set(ttHandle, 'String', sprintf('t=%d', k));

                drawnow;
                pause(pausetime);
            end
        end
    end

    tt = t;

    if doplot
        kSnapshot = round(T/2);

        nGrid = 9;
        xVals = linspace(-0.5, 12, nGrid);
        yVals = linspace(-0.5, 12, nGrid);
        [Xg, Yg] = meshgrid(xVals, yVals);
        DX = zeros(size(Xg));
        DY = zeros(size(Xg));

        ExVal = Ex(kSnapshot);
        EyVal = Ey(kSnapshot);

        for i = 1:numel(Xg)
            x_model = Xg(i) / w;
            y_model = Yg(i) / w;

            sx = -w*y_model + lambda*ExVal;
            sy = -w*x_model + lambda*EyVal;

            dx = -x_model + iofunc(sx, a, b, c);
            dy = -y_model + iofunc(sy, a, b, c);

            DX(i) = dx*w*10;
            DY(i) = dy*w*10;
        end

        hold on;
        quiver(Xg, Yg, DX, DY, 'Color', 'b', ...
               'AutoScale', 'on', 'AutoScaleFactor', 1);
    end

    box off

    function y = iofunc(s, a, b, c)
        y = a * tanh(s + b) + c;
    end
end
