function [fps_stable, nStable, fps_all, isStable] = count_fixed_points_offer(params, f1, f2)

    % Initialize outputs so they always exist
    fps_stable = [];
    fps_all    = [];
    isStable   = [];
    nStable    = 0;

    % Same model equations as simulation / plotting code
    ExVal = params.alphareward * f1 - params.alphapunish * f2;
    EyVal = 0;

    a = params.a;
    b = params.b;
    c = params.c;
    w = params.w_i;
    lambda = params.lambda;

    % Nullclines
    Fx = @(y) a .* tanh(-w .* y + lambda .* ExVal + b) + c;   % x = Fx(y)
    Gy = @(x) a .* tanh(-w .* x + lambda .* EyVal + b) + c;   % y = Gy(x)

    % Composed 1D fixed-point equation
    H = @(x) Fx(Gy(x)) - x;

    % Search range
    xmin = -5;
    xmax = 80;
    xgrid = linspace(xmin, xmax, 10000);
    hvals = H(xgrid);

    roots_x = [];

    % ------------------------------------------------------------
    % 1) Standard sign-change roots
    % ------------------------------------------------------------
    for k = 1:numel(xgrid)-1
        x1 = xgrid(k);
        x2 = xgrid(k+1);
        h1 = hvals(k);
        h2 = hvals(k+1);

        if ~isfinite(h1) || ~isfinite(h2)
            continue
        end

        if h1*h2 < 0
            try
                xr = fzero(H, [x1 x2]);
                if isfinite(xr)
                    roots_x(end+1,1) = xr; %#ok<AGROW>
                end
            catch
            end
        end
    end

    % ------------------------------------------------------------
    % 2) Near-zero local minima to catch tangent roots
    % ------------------------------------------------------------
    absH = abs(hvals);
    tol_zero = 1e-4;

    for k = 2:numel(xgrid)-1
        if ~isfinite(absH(k-1)) || ~isfinite(absH(k)) || ~isfinite(absH(k+1))
            continue
        end

        isLocalMin = absH(k) <= absH(k-1) && absH(k) <= absH(k+1);

        if isLocalMin && absH(k) < tol_zero
            xa = xgrid(k-1);
            xb = xgrid(k+1);

            try
                xr = fminbnd(@(x) abs(H(x)), xa, xb);
                if isfinite(xr) && abs(H(xr)) < tol_zero
                    roots_x(end+1,1) = xr; %#ok<AGROW>
                end
            catch
            end
        end
    end

    % ------------------------------------------------------------
    % 3) Build all fixed points
    % ------------------------------------------------------------
    for i = 1:numel(roots_x)
        xr = roots_x(i);
        yr = Gy(xr);

        if isfinite(xr) && isfinite(yr)
            fps_all(end+1,:) = [xr yr]; %#ok<AGROW>
        end
    end

    % Deduplicate
    if ~isempty(fps_all)
        fps_all = sortrows(fps_all,1);
        keep = true(size(fps_all,1),1);
        tol_dup = 1e-2;

        for i = 2:size(fps_all,1)
            if norm(fps_all(i,:) - fps_all(i-1,:)) < tol_dup
                keep(i) = false;
            end
        end
        fps_all = fps_all(keep,:);
    end

    % ------------------------------------------------------------
    % 4) Stability classification
    % ------------------------------------------------------------
    isStable = false(size(fps_all,1),1);

    for i = 1:size(fps_all,1)
        x = fps_all(i,1);
        y = fps_all(i,2);

        dFx_dy = a * (1 - tanh(-w*y + lambda*ExVal + b).^2) * (-w);
        dGy_dx = a * (1 - tanh(-w*x + lambda*EyVal + b).^2) * (-w);

        J = [-1,      dFx_dy;
             dGy_dx, -1];

        ev = eig(J);
        isStable(i) = all(real(ev) < 0);
    end

    % Final outputs
    nStable = sum(isStable);
    fps_stable = fps_all(isStable,:);
end