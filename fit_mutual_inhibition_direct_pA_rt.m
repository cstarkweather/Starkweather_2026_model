function fitout = fit_mutual_inhibition_direct_pA_rt(target, offers, simcfg)

    clear_anchor_cache();

    % ----- use a priori bounds from simcfg -----
    lb = simcfg.lb(:)';
    ub = simcfg.ub(:)';

    % parameter order must match unpack_dyn:
    % [alphareward alphapunish noise w_i a lambda b c time_stable thresh]

    % ----- explicit hand start (kept if inside bounds) -----
    x_hand = [ ...
        0.08 ...  % alphareward
        0.13 ...  % alphapunish
        80   ...  % noise
        0.08 ...  % w_i
        22   ...  % a
        2.5  ...  % lambda
        0.5  ...  % b
        22   ...  % c
        400  ...  % time_stable
        0.90 ...  % thresh
    ];

    if any(x_hand < lb) || any(x_hand > ub)
        warning('x_hand is outside simcfg bounds; clipping to bounds.');
        x_hand = min(max(x_hand, lb), ub);
    end

    % ----- sample candidate starts -----
    nCandidates = 1000;
    nParams     = numel(lb);

    Xcand = rand(nCandidates, nParams);
    Xcand = lb + Xcand .* (ub - lb);
    Xcand(1,:) = x_hand;

    % enforce physiological start condition c >= a (no negative neural activity)
Xcand(:,8) = max(Xcand(:,8), Xcand(:,5));
x_hand(8)  = max(x_hand(8), x_hand(5));
Xcand(1,:) = x_hand;

    % ----- prescreen storage -----
    keepMask = false(nCandidates,1);
    keepInfo = cell(nCandidates,1);

    % ----- prescreen starts using low-conflict anchors only -----
    % retain only if:
    %   at least one stable fixed point in approach anchor
    %   at least one stable fixed point in avoid anchor
    %   best approach-like stable point has x-y >= 0.5
    %   best avoid-like stable point has x-y <= -0.5

    nonlcon = @nonlincon_phys;

    for s = 1:nCandidates
        x0 = Xcand(s,:);

        try
            info = prescreen_anchor_start(x0, offers, simcfg);
            keepInfo{s} = info;

            isOK = ...
                info.nStable_appr    >= 1   && ...
                info.nStable_avoid   >= 1   && ...
                info.best_appr_bias  >= 0.5 && ...
                info.best_avoid_bias <= -0.5;

            keepMask(s) = isOK;

        catch ME
            keepMask(s) = false;
            fprintf('Prescreen failed at candidate %d: %s\n', s, ME.message);
        end
    end

    % ----- prescreen diagnostics -----
    best_appr_bias_all  = nan(nCandidates,1);
    best_avoid_bias_all = nan(nCandidates,1);
    traj_appr_all       = nan(nCandidates,1);
    traj_avoid_all      = nan(nCandidates,1);
    nStable_appr_all    = nan(nCandidates,1);
    nStable_avoid_all   = nan(nCandidates,1);
    act_all             = nan(nCandidates,1);
    noise_all           = nan(nCandidates,1);

    for s = 1:nCandidates
        if ~isempty(keepInfo{s})
            info = keepInfo{s};

            if isfield(info,'anchor_activity'),  act_all(s) = info.anchor_activity; end
            if isfield(info,'best_appr_bias'),   best_appr_bias_all(s) = info.best_appr_bias; end
            if isfield(info,'best_avoid_bias'),  best_avoid_bias_all(s) = info.best_avoid_bias; end
            if isfield(info,'traj_bias_appr'),   traj_appr_all(s) = info.traj_bias_appr; end
            if isfield(info,'traj_bias_avoid'),  traj_avoid_all(s) = info.traj_bias_avoid; end
            if isfield(info,'nStable_appr'),     nStable_appr_all(s) = info.nStable_appr; end
            if isfield(info,'nStable_avoid'),    nStable_avoid_all(s) = info.nStable_avoid; end
            if isfield(info,'noise_ratio'),      noise_all(s) = info.noise_ratio; end
        end
    end

    fprintf('anchor_activity:   median %.3f, 95th %% %.3f\n', ...
        median(act_all,'omitnan'), prctile(act_all,95));

    fprintf('best_appr_bias:    median %.3f, 95th %% %.3f\n', ...
        median(best_appr_bias_all,'omitnan'), prctile(best_appr_bias_all,95));

    fprintf('best_avoid_bias:   median %.3f, 5th %% %.3f\n', ...
        median(best_avoid_bias_all,'omitnan'), prctile(best_avoid_bias_all,5));

    fprintf('traj_bias_appr:    median %.3f, 95th %% %.3f\n', ...
        median(traj_appr_all,'omitnan'), prctile(traj_appr_all,95));

    fprintf('traj_bias_avoid:   median %.3f, 5th %% %.3f\n', ...
        median(traj_avoid_all,'omitnan'), prctile(traj_avoid_all,5));

    fprintf('nStable_appr:      median %.3f, 95th %% %.3f\n', ...
        median(nStable_appr_all,'omitnan'), prctile(nStable_appr_all,95));

    fprintf('nStable_avoid:     median %.3f, 95th %% %.3f\n', ...
        median(nStable_avoid_all,'omitnan'), prctile(nStable_avoid_all,95));

    fprintf('noise_ratio:       median %.3f, 5th-95th [%.3f %.3f]\n', ...
        median(noise_all,'omitnan'), prctile(noise_all,5), prctile(noise_all,95));

    % ----- keep all passing starts -----
    idxKeep = find(keepMask);
    nPassed = numel(idxKeep);

    fprintf('Prescreen accepted %d / %d candidates.\n', nPassed, nCandidates);
    fprintf('Prescreen rejected %d / %d candidates.\n', nCandidates - nPassed, nCandidates);

    if isempty(idxKeep)
        error('No starts passed prescreen. Relax thresholds or increase nCandidates.');
    end

    X0 = Xcand(idxKeep,:);
    fprintf('Optimizing all %d accepted starts.\n', size(X0,1));

    % ----- objective -----
    obj = @(x) obj_direct_fit_pA_rt(x, target, offers, simcfg);

    % ----- optimizer options -----
    opts = optimoptions('fmincon', ...
        'Display','iter', ...
        'Algorithm','interior-point', ...
        'MaxIterations',80, ...
        'MaxFunctionEvaluations',400, ...
        'FiniteDifferenceStepSize',1e-2, ...
        'StepTolerance',1e-4, ...
        'OptimalityTolerance',1e-4);

    % ----- multistart optimization -----
    best.err  = inf;
    best.xhat = [];
    best.pred = struct();
    best.fval = inf;
    best.i0   = [];

    allfits = struct([]);

    for s = 1:size(X0,1)
        x0 = X0(s,:);

        fprintf('\n===== Optimizing start %d / %d =====\n', s, size(X0,1));

        try
            f0 = obj(x0);

            if ~isfinite(f0) || ~isreal(f0)
                warning('Invalid initial objective at start %d', s);
                allfits(s).x0    = x0;
                allfits(s).xhat  = nan(size(x0));
                allfits(s).fval  = nan;
                allfits(s).err   = nan;
                allfits(s).pred  = struct();
                allfits(s).error = 'Invalid initial objective';
                continue
            end

            %[xhat, fval] = fmincon(obj, x0, [], [], [], [], lb, ub, [], opts);
            
            [xhat, fval] = fmincon(obj, x0, [], [], [], [], lb, ub, nonlcon, opts);
            [err, pred]  = obj_direct_fit_pA_rt(xhat, target, offers, simcfg);

            allfits(s).x0   = x0;
            allfits(s).xhat = xhat;
            allfits(s).fval = fval;
            allfits(s).err  = err;
            allfits(s).pred = pred;

            if isfinite(err) && err < best.err
                best.err  = err;
                best.xhat = xhat;
                best.pred = pred;
                best.fval = fval;
                best.i0   = s;
            end

        catch ME
            warning('Start %d failed: %s', s, ME.message);

            allfits(s).x0    = x0;
            allfits(s).xhat  = nan(size(x0));
            allfits(s).fval  = nan;
            allfits(s).err   = nan;
            allfits(s).pred  = struct();
            allfits(s).error = ME.message;
        end
    end

    if isempty(best.xhat)
        error('All optimization starts failed.');
    end

    % ----- final output -----
    fitout.xhat        = best.xhat;
    fitout.fval        = best.fval;
    fitout.err         = best.err;
    fitout.pred        = best.pred;
    fitout.params      = unpack_dyn(best.xhat);
    fitout.bestStartIx = best.i0;
    fitout.allfits     = allfits;

    fitout.prescreen.Xcand    = Xcand;
    fitout.prescreen.keepMask = keepMask;
    fitout.prescreen.keepInfo = keepInfo;
    fitout.prescreen.idxKeep  = idxKeep;

    % ----- plot -----
    figure('Color','w');
    plot(target.pA_mean, 'ko-', 'LineWidth', 1.5); hold on
    plot(best.pred.pA, 'r.-', 'MarkerSize', 18)
    xlabel('trial type')
    ylabel('p(approach)')
    legend({'human mean','model'}, 'Location','best')
    title('Choice fit (prescreened multistart)')
    box off
    set(gca,'TickDir','out')

    disp('Final fitted parameters:');
    disp(fitout.params);

    disp('Final objective value:');
    disp(fitout.err);
end

function [cineq, ceq] = nonlincon_phys(x)
% Enforce nonnegative transfer function output:
% iofunc(s) = a*tanh(s+b) + c >= 0 for all s
%
% Since tanh(s+b) is in [-1,1], the minimum output is c - a
% assuming a >= 0. So require c - a >= 0, i.e. a - c <= 0.

    p = unpack_dyn(x);

    cineq = p.a - p.c;   % must be <= 0
    ceq   = [];
end