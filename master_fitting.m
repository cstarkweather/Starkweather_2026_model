clear
clear functions
clc

load('behavior_data.mat')

% -------------------- OFFERS --------------------
offers11 = [ ...
    2 0;
    2 2;
    2 4;
    4 0;
    4 2;
    4 4;
    7 0;
    7 2;
    7 4;
    0 2;
    0 4];

offers.reward = offers11(:,1)';
offers.punish = offers11(:,2)';

% -------------------- TARGET + FIT --------------------
target = compute_target_behavior_offers11_with_conflict(behavior_data, offers11);

simcfg = struct();
simcfg.anchor_appr_type  = 7;
simcfg.anchor_avoid_type = 11;
simcfg.smooth_post       = 1;
simcfg.rngSeed           = 1;
simcfg.n_trials_sim      = 25;
simcfg.anchor_trials     = 8;
simcfg.burnin            = 500;

% bounds for tested parameters
simcfg.lb = [ ...
    0 ...  % alphareward 
    0 ...  % alphapunish 
    0  ...  % noise
    0.02 ...  % w_i
    1  ...  % a 
    0.5  ...  % lambda
   0    ...  % b
   1     ...  % c
   100    ...  % time_stable
    0.6 ...  % thresh
];

simcfg.ub = [ ...
    1    ...  % alphareward
    1    ...  % alphapunish
    100   ...  % noise
    0.6    ...  % w_i
    30   ...  % a
    5   ...  % lambda
    3    ...  % b
    30   ...  % c
   600  ...  % time_stable
    0.9 ...  % thresh
];

fit1 = fit_mutual_inhibition_direct_pA_rt(target, offers, simcfg);

save('fit_results.mat','-v7.3')

%% plot nullclines for 3 trial types using fitted parameters

p = fit1.params;

% extract errors
errs = arrayfun(@(s) s.err, fit1.allfits);

% sort (ascending = best first)
[errs_sorted, idx] = sort(errs, 'ascend');

% get 5th best
% fit5 = fit1.allfits(idx(5));
% p = unpack_dyn(fit5.xhat);
%p = x0;
%temp = fit1.allfits(5).x0;
%p = unpack_dyn(fit_hand.xhat);
combos = [0 4; 7 4; 7 0];   % [reward punishment]


figure('Color','w');
for i = 1:size(combos,1)
    subplot(1,3,i)

    mfsim_fitting_plot_nullclines_figure( ...
        combos(i,1), ...   % reward
        combos(i,2), ...   % punishment
        0.001, ...         % pause time
        1, ...             % doplot
        p.alphareward, ...
        p.alphapunish, ...
        p.noise, ...
        p.lambda, ...
        p.w_i, ...
        p.a, ...
        p.b, ...
        p.c, ...
        1);                % visualize_trajectory

    title(sprintf('R=%g, P=%g', combos(i,1), combos(i,2)));
end


%% -------------------- SESSION SIMULATION SETTINGS --------------------
sessioncfg = struct();
sessioncfg.n_trials_session = 220;
sessioncfg.burnin           = simcfg.burnin;
sessioncfg.smooth_post      = simcfg.smooth_post;
clear functions
clear_anchor_cache

% posterior classifier from anchors using fitted params
post = get_anchor_posterior_from_params(p, offers, simcfg);

rng('shuffle')
S = simulate_session_220(p, offers, post, sessioncfg);
% ---------------------------------
% Per-trial-type summaries
% ---------------------------------
nTypes = numel(S.trial_type);

avg_num_states = nan(1, nTypes);
avg_states_per_sec = nan(1, nTypes);
avg_decision_time_ms = nan(1, nTypes);

for i = 1:nTypes
    ns = S.trial_type(i).num_states(:);
    dt = S.trial_type(i).decision_times(:);   % ms

    avg_num_states(i) = mean(ns, 'omitnan');
    avg_decision_time_ms(i) = mean(dt, 'omitnan');

    valid = ~isnan(ns) & ~isnan(dt) & dt > 0;
    avg_states_per_sec(i) = mean(ns(valid) ./ (dt(valid)/1000), 'omitnan');
end

conflict = S.conflict(:)';

nTypes = numel(S.trial_type);

avg_num_states = nan(1, nTypes);
avg_states_per_sec = nan(1, nTypes);

for i = 1:nTypes
    ns = S.trial_type(i).num_states(:);
    dt = S.trial_type(i).decision_times(:);   % ms

    avg_num_states(i) = mean(ns, 'omitnan');

    valid = ~isnan(ns) & ~isnan(dt) & dt > 0;
    avg_states_per_sec(i) = mean(ns(valid) ./ (dt(valid)/1000), 'omitnan');
end

conflict = S.conflict(:)';
S.p_approach
% plot model behavior figure
figure('Color','w');

subplot(1,2,1)
plot(conflict, avg_num_states, 'k.', 'MarkerSize', 22);
hold on;
[rho1, pval1] = corr(conflict(:), avg_num_states(:), ...
    'Type','Spearman', 'Rows','complete');
pfit1 = polyfit(conflict(:), avg_num_states(:), 1);
xfit1 = linspace(min(conflict), max(conflict), 100);
plot(xfit1, polyval(pfit1, xfit1), 'k-', 'LineWidth', 2);
box off;
set(gca,'TickDir','out');
xlabel('Conflict');
ylabel('Avg number of states');
title(sprintf('\\rho = %.3f, p = %.3g', rho1, pval1));
xlim([0 1]);

subplot(1,2,2)
plot(conflict, avg_states_per_sec, 'k.', 'MarkerSize', 22);
hold on;
[rho2, pval2] = corr(conflict(:), avg_states_per_sec(:), ...
    'Type','Spearman', 'Rows','complete');
pfit2 = polyfit(conflict(:), avg_states_per_sec(:), 1);
xfit2 = linspace(min(conflict), max(conflict), 100);
plot(xfit2, polyval(pfit2, xfit2), 'k-', 'LineWidth', 2);
box off;
set(gca,'TickDir','out');
xlabel('Conflict');
ylabel('Avg number of states / second');
title(sprintf('\\rho = %.3f, p = %.3g', rho2, pval2));
xlim([0 1]);


%% % build PSTHs for + and - nodes
% =========================================================
% PSTHs from session struct S
% 2000 ms prior to decision, with NaN left-padding
%
% Assumption:
%   S.trajectory_x = (+) node
%   S.trajectory_y = (-) node
%
% If reversed, swap trajx and trajy below.
% =========================================================
figure()
window_ms = 5000;
smoothing = 20;

apr_pos = [];   % (+) node, approach trials
apr_neg = [];   % (-) node, approach trials
av_pos  = [];   % (+) node, avoid trials
av_neg  = [];   % (-) node, avoid trials

nTrials = numel(S.decision);

for tr = 1:nTrials

    dt = round(S.decision_time(tr));

    if isnan(dt)
        continue
    end
    

    trajx = S.trajectory_x{tr};
    trajy = S.trajectory_y{tr};

    if isempty(trajx) || isempty(trajy)
        continue
    end

    % make sure we do not index past available samples
    T = min([numel(trajx), numel(trajy), dt]);

    % take as much pre-decision trajectory as exists
    start_idx = max(1, T - window_ms + 1);
    pos_trace = trajx(start_idx:T);   % (+) node
    neg_trace = trajy(start_idx:T);   % (-) node

    % left-pad with NaNs so every row is exactly 2000 samples
    pad_len = window_ms - numel(pos_trace);
    pos_trace = [nan(1,pad_len), pos_trace];
    neg_trace = [nan(1,pad_len), neg_trace];

    if S.decision(tr) == 1
        apr_pos = [apr_pos; pos_trace];
        apr_neg = [apr_neg; neg_trace];
    elseif S.decision(tr) == 0
        av_pos  = [av_pos; pos_trace];
        av_neg  = [av_neg; neg_trace];
    end
end

% sanity checks
disp(size(apr_pos))
disp(size(av_pos))
disp(size(apr_neg))
disp(size(av_neg))

dt = round(S.decision_time(tr));



% =========================================================
% Mean and SEM
% =========================================================
mean_apr_pos = mean(apr_pos, 1, 'omitnan');
sem_apr_pos  = std(apr_pos, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(apr_pos),1));

mean_av_pos  = mean(av_pos, 1, 'omitnan');
sem_av_pos   = std(av_pos, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(av_pos),1));

mean_apr_neg = mean(apr_neg, 1, 'omitnan');
sem_apr_neg  = std(apr_neg, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(apr_neg),1));

mean_av_neg  = mean(av_neg, 1, 'omitnan');
sem_av_neg   = std(av_neg, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(av_neg),1));

% =========================================================
% Smooth
% =========================================================
sm_mean_apr_pos = smooth(mean_apr_pos, smoothing);
sm_sem_apr_pos  = smooth(sem_apr_pos,  smoothing);

sm_mean_av_pos  = smooth(mean_av_pos, smoothing);
sm_sem_av_pos   = smooth(sem_av_pos,  smoothing);

sm_mean_apr_neg = smooth(mean_apr_neg, smoothing);
sm_sem_apr_neg  = smooth(sem_apr_neg,  smoothing);

sm_mean_av_neg  = smooth(mean_av_neg, smoothing);
sm_sem_av_neg   = smooth(sem_av_neg,  smoothing);

% =========================================================
% Time axis
% =========================================================
xvals = -window_ms+1:0;


% Force all plotting vectors to row shape
xvals = xvals(:)';

sm_mean_apr_neg = sm_mean_apr_neg(:)';
sm_sem_apr_neg  = sm_sem_apr_neg(:)';
sm_mean_av_neg  = sm_mean_av_neg(:)';
sm_sem_av_neg   = sm_sem_av_neg(:)';

sm_mean_apr_pos = sm_mean_apr_pos(:)';
sm_sem_apr_pos  = sm_sem_apr_pos(:)';
sm_mean_av_pos  = sm_mean_av_pos(:)';
sm_sem_av_pos   = sm_sem_av_pos(:)';


% -------------------------
% (-) node
% -------------------------
subplot(1,2,1);
hold on

apr_upper = sm_mean_apr_neg + sm_sem_apr_neg;
apr_lower = sm_mean_apr_neg - sm_sem_apr_neg;
fill([xvals fliplr(xvals)], ...
     [apr_upper fliplr(apr_lower)], ...
     'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
plot(xvals, sm_mean_apr_neg, 'b-', 'LineWidth', 1.5);

av_upper = sm_mean_av_neg + sm_sem_av_neg;
av_lower = sm_mean_av_neg - sm_sem_av_neg;
fill([xvals fliplr(xvals)], ...
     [av_upper fliplr(av_lower)], ...
     'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
plot(xvals, sm_mean_av_neg, 'r-', 'LineWidth', 1.5);

xline(0, 'k--');
xlabel('Time from decision (ms)');
ylabel('Mean activity');
title('(-) node');
xlim([-2000 0]);
box off
set(gca, 'TickDir', 'out');

% -------------------------
% (+) node
% -------------------------
subplot(1,2,2);
hold on

apr_upper = sm_mean_apr_pos + sm_sem_apr_pos;
apr_lower = sm_mean_apr_pos - sm_sem_apr_pos;
fill([xvals fliplr(xvals)], ...
     [apr_upper fliplr(apr_lower)], ...
     'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
plot(xvals, sm_mean_apr_pos, 'b-', 'LineWidth', 1.5);

av_upper = sm_mean_av_pos + sm_sem_av_pos;
av_lower = sm_mean_av_pos - sm_sem_av_pos;
fill([xvals fliplr(xvals)], ...
     [av_upper fliplr(av_lower)], ...
     'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
plot(xvals, sm_mean_av_pos, 'r-', 'LineWidth', 1.5);

xline(0, 'k--');
xlabel('Time from decision (ms)');
ylabel('Mean activity');
title('(+) node');
xlim([-2000 0]);
box off
set(gca, 'TickDir', 'out');


%% Prospect theory model
[decision, reaction_time, reward_trial_type, punishment_trial_type, ...
    p_approach_trial_type, conflict_trial_type, reward_trial, punish_trial] = ...
    build_behavior_vectors_from_subject(S, offers.reward, offers.punish);

conflict_trial = S.conflict_trial(:);

conflict_trial_type = zeros(size(decision));

for i = 1:11
    idx = reward_trial_type == offers.reward(i) & ...
          punishment_trial_type == offers.punish(i);

    conflict_trial_type(idx) = S.conflict(i);
end

figure('Color','w','Position',[100 100 1800 300]);

[rho_conflict, pval_conflict, rho_rt, pval_rt, ...
 alpha_reward, alpha_punish, lambda_fit, mu_fit] = ...
    plotSubjectData(1, NaN, NaN, ...
                    reward_trial, punish_trial, ...
                    decision, conflict_trial_type, ...
                    reaction_time, reward_trial_type, ...
                    punishment_trial_type, p_approach_trial_type, ...
                    conflict_trial_type);


%% Adapted plotSubjectData function using prospect theory model
function [rho_conflict, pval_conflict, rho_rt, pval_rt, alpha_reward_opt, alpha_punish_opt, lambda_opt, mu_opt] = ...
    plotSubjectData(subjectID, reward_coeff, punish_coeff, rewards_trials, punishments_trials, ...
                    decision, conflict_all, reaction_time_all, reward_trial_types, punishment_trial_types, p_approach, conflict_trial)

    filter = find(decision < 2);

    rewards = rewards_trials(filter);
    punishments = punishments_trials(filter);
    decisions = decision(filter);
    reaction_times = reaction_time_all(filter);

    reward_sizes = 0:1:7;
    punishment_sizes = 0:1:4;

    mat = NaN(length(punishment_sizes), length(reward_sizes));
    for i = 1:length(punishment_sizes)
        for j = 1:length(reward_sizes)
            idx = find(punishment_trial_types == punishment_sizes(i) & ...
                       reward_trial_types == reward_sizes(j));
            if ~isempty(idx)
                mat(i, j) = mean(p_approach(idx), 'omitnan');
            end
        end
    end

    [X_grid, Y_grid] = meshgrid(reward_sizes, punishment_sizes);
    X_flat = X_grid(:);
    Y_flat = Y_grid(:);
    P_flat = mat(:);

    valid_idx = ~isnan(P_flat);
    X_flat = X_flat(valid_idx);
    Y_flat = Y_flat(valid_idx);
    P_flat = P_flat(valid_idx);

    p_reward = 0.42;
    p_punishment = 0.4;

    objfun = @(params) costfun(params, X_flat, Y_flat, P_flat, p_reward, p_punishment);
    init_params = [1, 1, 1, 1];
    options = optimset('Display','off','TolFun',1e-8);
    [opt_params, ~] = fminsearch(objfun, init_params, options);

    mu_opt = opt_params(1);
    alpha_reward_opt = opt_params(2);
    alpha_punish_opt = opt_params(3);
    lambda_opt = opt_params(4);

    V_grid = p_reward * (X_grid.^alpha_reward_opt) - ...
             p_punishment * lambda_opt * (Y_grid.^alpha_punish_opt);

    subplot(1,6,2)
    imagesc(reward_sizes, punishment_sizes, V_grid)
    colormap bone
    colorbar
    hold on
    contour(reward_sizes, punishment_sizes, V_grid, [0 0], 'LineColor', 'k', 'LineWidth', 2);
    set(gca, 'YDir', 'normal');
    xlabel('Reward size');
    ylabel('Punishment size');
    title('Decision boundary (V=0)');
    box off
    set(gca, 'TickDir', 'out');
    hold off

    subplot(1,6,1)
    [x_sw, y_sw] = beeswarm2D_on_grid(rewards, punishments, 0.5);
    scatter(x_sw(decisions==1), y_sw(decisions==1), 50, 'o', 'MarkerEdgeColor', 'b'); hold on
    scatter(x_sw(decisions==0), y_sw(decisions==0), 50, '+', 'MarkerEdgeColor', 'r');
    contour(reward_sizes, punishment_sizes, V_grid, [0 0], 'LineColor', 'k', 'LineWidth', 2);
    xlim([-0.5 7.5]); ylim([-0.5 4.5]);
    xlabel('Reward size'); ylabel('Punishment size');
    title('Trial data & decision boundary');
    box off
    set(gca, 'TickDir', 'out');
    hold off

% SUBPLOT 3: Trial conflict vs. |Value|
% SUBPLOT 3: Condition entropy conflict vs. |Value|
subplot(1, 6, 3)

reward_by_type = unique(reward_trial_types(:), 'stable');
punish_by_type = unique(punishment_trial_types(:), 'stable'); %#ok<NASGU>
reward_by_type = [2 2 2 4 4 4 7 7 7 0 0]';
punish_by_type = [0 2 4 0 2 4 0 2 4 2 4]';

Value_type = p_reward * (reward_by_type.^alpha_reward_opt) - ...
             p_punishment * lambda_opt * (punish_by_type.^alpha_punish_opt);

% one conflict value per offer type
conflict_by_type = nan(size(Value_type));
for k = 1:numel(Value_type)
    idx = reward_trial_types == reward_by_type(k) & ...
          punishment_trial_types == punish_by_type(k);
    conflict_by_type(k) = mean(conflict_all(idx), 'omitnan');
end

plot(abs(Value_type), conflict_by_type, 'k.', 'MarkerSize', 20);
hold on;

[rho_conflict, pval_conflict] = corr(abs(Value_type), conflict_by_type, ...
    'Type', 'Spearman', 'Rows', 'complete');

pfit = polyfit(abs(Value_type), conflict_by_type, 1);
x_fit = linspace(min(abs(Value_type)), max(abs(Value_type)), 100);
plot(x_fit, polyval(pfit, x_fit), 'r-', 'LineWidth', 2);

text(max(x_fit), max(conflict_by_type), ...
    sprintf('p = %.3f, rho = %.3f', pval_conflict, rho_conflict), ...
    'FontSize', 12, 'VerticalAlignment', 'top', 'HorizontalAlignment', 'right');

xlabel('|Decision Value|');
ylabel('Condition entropy conflict');
title('Conflict vs. |Value|');
box off;
set(gca, 'TickDir', 'out');
hold off;

    subplot(1,6,4)
    Val_apr_rt = p_reward * (rewards.^alpha_reward_opt) - ...
                 p_punishment * lambda_opt * (punishments.^alpha_punish_opt);
    decision_function_rt = abs(Val_apr_rt);
    scatter(decision_function_rt + 0.01*randn(length(decision_function_rt),1), reaction_times, 5, 'k.');
    hold on
    [rho_rt, pval_rt] = corr(decision_function_rt(:), reaction_times(:), 'Type','Spearman');
    mdl_rt = fitlm(decision_function_rt, reaction_times);
    x_fit_rt = linspace(min(decision_function_rt), max(decision_function_rt), 100);
    y_fit_rt = mdl_rt.Coefficients.Estimate(1) + mdl_rt.Coefficients.Estimate(2)*x_fit_rt;
    plot(x_fit_rt, y_fit_rt, 'r-', 'LineWidth', 2);
    text(max(x_fit_rt), max(y_fit_rt), sprintf('p = %.3f, rho = %.3f', pval_rt, rho_rt), ...
        'FontSize', 12, 'VerticalAlignment', 'top', 'HorizontalAlignment', 'right');
    xlabel('|Decision Value|'); ylabel('Reaction time (s)');
    title('RT vs. |Value|');
    box off
    set(gca, 'TickDir', 'out');
    hold off

% SUBPLOT 5: Trial conflict vs. Value
subplot(1, 6, 5)
% SUBPLOT 5: Condition entropy conflict vs. Value
subplot(1, 6, 5)

reward_by_type = [2 2 2 4 4 4 7 7 7 0 0]';
punish_by_type = [0 2 4 0 2 4 0 2 4 2 4]';

Value_type = p_reward * (reward_by_type.^alpha_reward_opt) - ...
             p_punishment * lambda_opt * (punish_by_type.^alpha_punish_opt);

conflict_by_type = zeros(size(Value_type));
for k = 1:numel(Value_type)
    idx = reward_trial_types == reward_by_type(k) & ...
          punishment_trial_types == punish_by_type(k);
    conflict_by_type(k) = mean(conflict_all(idx), 'omitnan');
end

plot(Value_type, conflict_by_type, 'k.', 'MarkerSize', 20);
hold on

[rho_conflict_signed, pval_conflict_signed] = corr(Value_type, conflict_by_type, ...
    'Type','Spearman', 'Rows','complete');

pfit = polyfit(Value_type, conflict_by_type, 1);
xfit = linspace(min(Value_type), max(Value_type), 100);
plot(xfit, polyval(pfit, xfit), 'r-', 'LineWidth', 2);

lim = max(abs(Value_type));
xlim([-lim-0.5 lim+0.5])

xlabel('Decision Value');
ylabel('Condition entropy conflict');
title(sprintf('Conflict vs. Value, \\rho = %.3f, p = %.3g', ...
    rho_conflict_signed, pval_conflict_signed));
box off
set(gca, 'TickDir', 'out');
hold off

    subplot(1,6,6)
    scatter(Val_apr_rt + 0.01*randn(length(Val_apr_rt),1), reaction_times, 5, 'k.');
    hold on
    xlabel('Decision Value'); ylabel('Reaction time (s)');
    xlim([-lim-0.5 lim+0.5])
    title('RT vs. Value');
    box off
    set(gca, 'TickDir', 'out');
    hold off
end

function err = costfun(params, X, Y, Pobs, p_reward, p_punishment)
    mu = params(1);
    alpha_reward = params(2);
    alpha_punish = params(3);
    lambda = params(4);

    V = p_reward * (X.^alpha_reward) - p_punishment * lambda * (Y.^alpha_punish);
    p_pred = 1 ./ (1 + exp(-mu * V));

    epsilon = 1e-10;
    p_pred = min(max(p_pred, epsilon), 1-epsilon);

    LL = sum(Pobs .* log(p_pred) + (1-Pobs) .* log(1-p_pred));
    err = -LL;
end

function [x_sw, y_sw] = beeswarm2D_on_grid(x, y, radius)
    x = x(:);
    y = y(:);
    x_sw = x;
    y_sw = y;

    cells = unique([x y], 'rows');

    step   = radius / 3;
    golden = pi * (3 - sqrt(5));

    for c = 1:size(cells,1)
        cx = cells(c,1);
        cy = cells(c,2);

        idx = find(x==cx & y==cy);
        n = numel(idx);
        if n <= 1
            continue
        end

        idx = sort(idx);

        for k = 1:n
            i = k - 1;
            if i == 0
                dx = 0; dy = 0;
            else
                r = step * sqrt(i);
                theta = i * golden;
                dx = r * cos(theta);
                dy = r * sin(theta);

                rr = hypot(dx, dy);
                if rr > radius
                    s = radius / rr;
                    dx = dx * s;
                    dy = dy * s;
                end
            end

            x_sw(idx(k)) = cx + dx;
            y_sw(idx(k)) = cy + dy;
        end
    end
end


