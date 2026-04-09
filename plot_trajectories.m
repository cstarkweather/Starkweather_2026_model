%% =========================================================
% Updated trajectory / posterior visualization using fit1
% Assumes fit1, offers, simcfg already exist in workspace
% =========================================================

rng(simcfg.rngSeed);   % reproducible example trajectory

p = fit1.params;
%% tanh function

% get parameters
if isfield(fit1, 'params') && ~isempty(fit1.params)
    p = fit1.params;
else
    % fallback if params not stored
    p = unpack_dyn(fit1.xhat);
end

a = p.a;
b = p.b;
c = p.c;

% input range
s = linspace(-10, 10, 1000);

% function
y = a * tanh(s + b) + c;

% plot
figure('Color','w'); hold on
plot(s, y, 'k-', 'LineWidth', 2)

% reference lines
xline(0, 'k--')
yline(c, 'r--', 'LineWidth', 1.5)

xlabel('Input s')
ylabel('Output y')
title(sprintf('IO function: a=%.2f, b=%.2f, c=%.2f', a, b, c))

box off
%% ---------------------------------------------------------
% 1) Nullclines for 3 trial types (same spirit as old plot)
% ---------------------------------------------------------
combos = [0 4; 7 4; 7 0];   % [reward punish]

figure('Color','w');
for ii = 1:size(combos,1)
    subplot(1,3,ii); hold on

    mfsim_fitting_plot_nullclines_fast( ...
        combos(ii,1), ...   % reward
        combos(ii,2), ...   % punishment
        0.001, ...
        1, ...              % doplot
        p.alphareward, ...
        p.alphapunish, ...
        p.noise, ...
        p.lambda, ...
        p.w_i, ... 
        p.a, ...
        p.b, ...
        p.c, ...
        1);                 % visualize_trajectory = 0 (nullclines only)

    title(sprintf('R=%d, P=%d', combos(ii,1), combos(ii,2)));
    axis square
    xlim([-0.5 8])
    ylim([-0.5 8])
    box off
    set(gca,'TickDir','out')
end


%% ---------------------------------------------------------
% 2) Get posterior model from the fitted best parameters
% ---------------------------------------------------------
% This assumes your current pipeline uses get_anchor_cache() to build the
% posterior model from the anchor trajectories.

post = get_anchor_cache(fit1.xhat, offers, simcfg);


%% ---------------------------------------------------------
% 3) Simulate one trajectory and color by state
%    state==1 (approach) = green
%    state==0 (avoid)    = red
% ---------------------------------------------------------
reward_val = 7;
punish_val = 4;

[trajx, trajy, tt] = mfsim_fitting_plot_nullclines_fast( ...
    reward_val, punish_val, ...
    0.001, ...
    0, ...                 % doplot off; we will make our own plot
    p.alphareward, ...
    p.alphapunish, ...
    p.noise, ...
    p.lambda, ...
    p.w_i, ...
    p.a, ...
    p.b, ...
    p.c, ...
    1);                    % visualize_trajectory on
%
% posterior over time for this trajectory
pAppr_t = compute_posterior_timeseries(trajx, trajy, post);
pAppr_t = pAppr_t(:);

% binary state
state = pAppr_t >= 0.5;    % 1 = approach, 0 = avoid

figure('Color','w'); hold on
subplot(2,1,1)
% draw nullclines underneath
mfsim_fitting_plot_nullclines_fast( ...
    reward_val, punish_val, ...
    0.001, ...
    1, ...
    p.alphareward, ...
    p.alphapunish, ...
    p.noise, ...
    p.lambda, ...
    p.w_i, ...
    p.a, ...
    p.b, ...
    p.c, ...
    0);

hold on

% 4) Alternative plot: same trajectory with time gradient
% ---------------------------------------------------------
cut = min(1000, numel(trajx));

xdata = smooth(trajx(1:cut), 20);
ydata = smooth(trajy(1:cut), 20);

%xdata = resample(xdata, 800, 1000);
%ydata = resample(ydata, 800, 1000);

N = numel(xdata);
alphaVal = 0.6;

c = [1-(1:N)'/N, (1:N)'/N, ones(N,1)];


scatter(xdata, ydata, 10, c, 'filled', ...
    'MarkerFaceAlpha', alphaVal, ...
    'MarkerEdgeAlpha', alphaVal);

xlabel('nS (plus neuron)')
ylabel('nS (minus neuron)')
title(sprintf('Time-gradient trajectory, R=%d P=%d', reward_val, punish_val))
axis equal
box off
set(gca,'TickDir','out')
xlim([-1.5 9])
ylim([-1.5 9])



% ---------------------------------------------------------
% 5) Posterior over time for the same time window as trajectory plot
%    colored with the same time gradient
% ---------------------------------------------------------

tplot = 1:cut;
pplot = pAppr_t(tplot);

% same color map as above
N = numel(tplot);
alphaVal = 0.6;
c = [1-(1:N)'/N, (1:N)'/N, ones(N,1)];
%
[decision_time, decision, num_states, num_pos_entries, num_neg_entries, entry_times] = ...
    detectStatePreferences(pAppr_t, round(p.time_stable), p.thresh);
decision_time
ax = subplot(2,1,2);
hold(ax, 'on')

% faint raw trace
plot(ax, tplot, pplot, '-', 'Color', [0.85 0.85 0.85], 'LineWidth', 1)

% colored points
scatter(ax, tplot, pplot, 12, c, 'filled', ...
    'MarkerFaceAlpha', alphaVal, ...
    'MarkerEdgeAlpha', alphaVal);

% smoothed trace
plot(ax, tplot, smooth(pplot, 50), 'k', 'LineWidth', 2)

% threshold
yline(ax, 0.5, '--', 'Color', [0.4 0.4 0.4])

% mark decision time if it occurred before cutoff
if ~isnan(decision_time) && decision_time >= 1 && decision_time <= cut
    xline(ax, decision_time, '--b', 'LineWidth', 1.5);

    % mark the posterior value at that time too
    scatter(ax, decision_time, pAppr_t(decision_time), 60, 'b', 'filled', ...
        'MarkerEdgeColor', 'k');
end

xlabel(ax, 'Time')
ylabel(ax, 'P(approach)')
title(ax, sprintf('Posterior over time, R=%d P=%d', reward_val, punish_val))
xlim(ax, [1 cut])
ylim(ax, [0 1])
box(ax, 'off')
set(ax, 'TickDir', 'out')

%% ---------------------------------------------------------
%% ---------------------------------------------------------
% Smoothed trajectory colored continuously by posterior
% using the same for-loop format as before
% pp = 0 -> red
% pp = 1 -> blue
% ---------------------------------------------------------

% draw nullclines underneath
mfsim_fitting_plot_nullclines_fast( ...
    reward_val, punish_val, ...
    0.001, ...
    1, ...
    p.alphareward, ...
    p.alphapunish, ...
    p.noise, ...
    p.lambda, ...
    p.w_i, ...
    p.a, ...
    p.b, ...
    p.c, ...
    0);

hold on

cut = min(1000, numel(trajx));

xplot = smooth(trajx(1:cut), 20);
yplot = smooth(trajy(1:cut), 20);
pp    = smooth(pAppr_t(1:cut), 20);

xplot = resample(xplot, 800, 1000);
yplot = resample(yplot, 800, 1000);
pp    = resample(pp,    800, 1000);

% make lengths match
xplot = xplot(:);
yplot = yplot(:);
pp    = pp(:);

n = min([numel(xplot), numel(yplot), numel(pp)]);
xplot = xplot(1:n);
yplot = yplot(1:n);
pp    = pp(1:n);

% clamp pp into [0,1]
pp(pp < 0) = 0;
pp(pp > 1) = 1;

for i = 1:numel(xplot)-1

    % pp=0 -> red, pp=1 -> blue
    col = [1-pp(i), 0, pp(i)];

    plot(xplot(i), yplot(i), '.','Markersize', 20,'Color', col);
end

% set matching colormap for colorbar
cmap = [linspace(1,0,256)', zeros(256,1), linspace(0,1,256)'];
colormap(cmap);
caxis([0 1]);

cb = colorbar;
cb.Label.String = 'P(Approach)';
cb.Ticks = [0 0.5 1];
cb.TickLabels = {'Avoid', '0.5', 'Approach'};

xlabel('nS (plus neuron)')
ylabel('nS (minus neuron)')
title('Smoothed trajectory colored by posterior')
axis equal
box off
set(gca,'TickDir','out')
xlim([-1.5 13.5])
ylim([-1.5 13.5])
%%
%% ---------------------------------------------------------
% Smoothed trajectory colored continuously by posterior
% fast version
% pp = 0 -> red
% pp = 1 -> blue
% ---------------------------------------------------------

% draw nullclines underneath
mfsim_fitting_plot_nullclines_fast( ...
    reward_val, punish_val, ...
    0.001, ...
    1, ...
    p.alphareward, ...
    p.alphapunish, ...
    p.noise, ...
    p.lambda, ...
    p.w_i, ...
    p.a, ...
    p.b, ...
    p.c, ...
    0);

hold on

cut = min(2700, numel(trajx));

xplot = smooth(trajx(1:cut), 20);
yplot = smooth(trajy(1:cut), 20);
pp    = smooth(pAppr_t(1:cut), 20);

xplot = resample(xplot, 800, 1000);
yplot = resample(yplot, 800, 1000);
pp    = resample(pp,    800, 1000);

% make lengths match
xplot = xplot(:);
yplot = yplot(:);
pp    = pp(:);

n = min([numel(xplot), numel(yplot), numel(pp)]);
xplot = xplot(1:n);
yplot = yplot(1:n);
pp    = pp(1:n);

% clamp pp into [0,1]
pp(pp < 0) = 0;
pp(pp > 1) = 1;

% build RGB colors from posterior
% pp=0 -> red, pp=1 -> blue
c = [1-pp, zeros(n,1), pp];

% one call = much faster
scatter(xplot, yplot, 20, c, 'filled');

% matching colorbar
cmap = [linspace(1,0,256)', zeros(256,1), linspace(0,1,256)'];
colormap(cmap);
caxis([0 1]);

cb = colorbar;
cb.Label.String = 'P(Approach)';
cb.Ticks = [0 0.5 1];
cb.TickLabels = {'Avoid', '0.5', 'Approach'};

xlabel('nS (plus neuron)')
ylabel('nS (minus neuron)')
title('Smoothed trajectory colored by posterior')
axis equal
box off
set(gca,'TickDir','out')
xlim([-0.5 13.5])
ylim([-0.5 13.5])
%%
colorbar