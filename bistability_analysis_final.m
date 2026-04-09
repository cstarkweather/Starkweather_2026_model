clear fp_counts fp_stable_counts fp_info ...
      fp_x0_counts fp_x0_stable_counts fp_x0_info ...
      errs idx errs_sorted

errs = arrayfun(@(s) s.err, fit1.allfits);
[errs_sorted, idx] = sort(errs, 'ascend');

R = offers.reward(:);
P = offers.punish(:);
nCond = numel(R);
nfits = numel(idx);

fp_counts        = nan(nfits, nCond);
fp_stable_counts = nan(nfits, nCond);
fp_info          = cell(nfits, nCond);

fp_x0_counts        = nan(nfits, nCond);
fp_x0_stable_counts = nan(nfits, nCond);
fp_x0_info          = cell(nfits, nCond);

for j = 1:nfits
    % optimized
    ph = unpack_dyn(fit1.allfits(idx(j)).xhat);

    for i = 1:nCond
        out = count_fixed_points_one_condition( ...
            R(i), P(i), ...
            ph.alphareward, ph.alphapunish, ...
            ph.lambda, ph.w_i, ph.a, ph.b, ph.c);

        fp_counts(j,i)        = out.nFixed;
        fp_stable_counts(j,i) = out.nStable;
        fp_info{j,i}          = out;
    end

    % initial
    if isfield(fit1.allfits, 'x0') && ~isempty(fit1.allfits(idx(j)).x0)
        p0 = unpack_dyn(fit1.allfits(idx(j)).x0);

        for i = 1:nCond
            out0 = count_fixed_points_one_condition( ...
                R(i), P(i), ...
                p0.alphareward, p0.alphapunish, ...
                p0.lambda, p0.w_i, p0.a, p0.b, p0.c);

            fp_x0_counts(j,i)        = out0.nFixed;
            fp_x0_stable_counts(j,i) = out0.nStable;
            fp_x0_info{j,i}          = out0;
        end
    end
end

nfit1s_gt11_xhat = sum(sum(fp_stable_counts, 2) > 11);
nfit1s_gt11_x0   = sum(sum(fp_x0_stable_counts, 2) > 11);
%%
delta_stable_total = sum(fp_stable_counts,2) - sum(fp_x0_stable_counts,2);

figure;
histogram(delta_stable_total,'Binwidth',0.1);
xlabel('\Delta total stable fixed points (xhat - x0)');
ylabel('Number of fit1s');
box off
%%
nfits = numel(fit1.allfits);

err_x0   = nan(nfits,1);
err_xhat = nan(nfits,1);

for s = 1:nfits
    x0   = fit1.allfits(s).x0;
    xhat = fit1.allfits(s).xhat;

    if isempty(x0) || isempty(xhat)
        continue
    end

    err_x0(s)   = obj_direct_fit_pA_rt(x0,   target, offers, simcfg);
    err_xhat(s) = obj_direct_fit_pA_rt(xhat, target, offers, simcfg);
end
%%
figure; hold on
plot(err_x0, err_xhat, 'k.')
plot([min(err_x0) max(err_x0)], [min(err_x0) max(err_x0)], 'r--')
xlabel('Error at x0')
ylabel('Error at xhat')
title('Optimization improvement')
axis square
xlim([0 5])
ylim([0 5])
%%
delta_err = err_x0 - err_xhat;
[err_xhat_sorted, idx] = sort(err_xhat, 'ascend');

delta_sorted = delta_err(idx);

figure; hold on

x = 1:numel(delta_sorted);


plot(x, delta_sorted, 'k.', 'MarkerSize', 14)
hold on
plot(x,err_xhat_sorted,'k')

yline(0, 'r--')

xlabel('fit1 rank (lowest → highest error at xhat)')
ylabel('\Delta error (x0 - xhat)')
title('Optimization improvement vs fit1 quality')

box off
set(gca,'TickDir','out')
ylim([0 3])

%%

% sorted error (already computed)
e_all = errs_sorted(:);

% stable points at highest-conflict offer (offer 9)
nStable9_all = fp_stable_counts(:,9);

% define valid fits (exclude degenerate tail)
% Option 1 (recommended): based on error plateau
valid = e_all < max(e_all(1:600)) ;   % replicates your cutoff behavior

% Option 2 (cleaner if you want):
% valid = abs(diff([e_all; e_all(end)])) > 1e-10;

% apply mask
e = e_all(valid);
nStable9 = nStable9_all(valid);

% binary bistability
is_bi = nStable9 >= 2;
%%
[b,~,stats] = glmfit(e, is_bi, 'binomial', 'link', 'logit');

beta = b(2);
p_logit = stats.p(2);

figure('Color','w'); hold on

% jittered scatter
ejit = e + 0.002*randn(size(e));
scatter(ejit, is_bi, 35, 'k', 'filled', ...
    'MarkerFaceAlpha', 0.5)

% logistic curve
xfit = linspace(min(e), max(e), 200);
yfit = glmval(b, xfit, 'logit');

plot(xfit, yfit, 'r-', 'LineWidth', 2)

xlabel('Model error (sorted)')
ylabel('P(bistable at offer 9)')
title(sprintf('Logistic slope = %.3f, p = %.3g', beta, p_logit))

ylim([-0.05 1.05])
box off
%%
group_mono = e(~is_bi);
group_bi   = e(is_bi);

window = [0 3];
binwidth = 0.1;

figure('Color','w')

% --- Monostable ---
subplot(2,1,1)
histogram(group_mono, ...
    'BinWidth', binwidth, ...
    'FaceColor', 'k', ...
    'EdgeColor', 'k')

xlim(window)
ylabel('Count')
title('Monostable')

set(gca, ...
    'TickDir','out', ...
    'Box','off', ...
    'LineWidth',1.2, ...
    'FontSize',12)

% --- Bistable ---
subplot(2,1,2)
histogram(group_bi, ...
    'BinWidth', binwidth, ...
    'FaceColor', 'k', ...
    'EdgeColor', 'k')

xlim(window)
xlabel('Model error')
ylabel('Count')
title('Bistable')

set(gca, ...
    'TickDir','out', ...
    'Box','off', ...
    'LineWidth',1.2, ...
    'FontSize',12)

% tighten spacing
set(gcf,'Position',[100 100 400 500])

p_rs = ranksum(group_mono, group_bi)

% stats
n_mono = numel(group_mono);
n_bi   = numel(group_bi);

med_mono = median(group_mono);
med_bi   = median(group_bi);

iqr_mono = iqr(group_mono);
iqr_bi   = iqr(group_bi);

% print everything
fprintf(['Logistic regression: beta = %.3f, p = %.3g\n' ...
         'Wilcoxon rank-sum: p = %.3g\n' ...
         'Monostable: n = %d, median = %.3f, IQR = %.3f\n' ...
         'Bistable:   n = %d, median = %.3f, IQR = %.3f\n'], ...
         beta, p_logit, ...
         p_rs, ...
         n_mono, med_mono, iqr_mono, ...
         n_bi,   med_bi,   iqr_bi);

fprintf(['Model error was associated with bistability (logistic slope = %.3f, p = %.3g). ' ...
         'Bistable solutions showed different error distributions compared to monostable solutions ' ...
         '(rank-sum p = %.3g; monostable median = %.3f [IQR %.3f], n = %d; ' ...
         'bistable median = %.3f [IQR %.3f], n = %d).\n'], ...
         beta, p_logit, ...
         p_rs, ...
         med_mono, iqr_mono, n_mono, ...
         med_bi,   iqr_bi,   n_bi);
%%
win = 20;

is_bi_sorted = is_bi;  % already sorted
p_bi = smooth(is_bi_sorted, win);

figure('Color','w')

yyaxis left
plot(e, 'k-', 'LineWidth', 1.5)
ylabel('Error')
ylim([0 3])   % <-- force probability axis

yyaxis right
plot(p_bi, 'r-', 'LineWidth', 2)
ylabel(sprintf('P(bistable), window=%d', win))
ylim([0 1])   % <-- force probability axis

xlabel('Fit rank (low → high error)')
title('Bistability enriched in low-error solutions')

set(gca, ...
    'TickDir','out', ...
    'Box','off', ...
    'LineWidth',1.2, ...
    'FontSize',12)
xlim([0 600])
%%
nCond = size(fp_stable_counts,2);

beta_all = nan(nCond,1);
p_all    = nan(nCond,1);

for i = 1:nCond
    is_bi_i = fp_stable_counts(valid,i) >= 2;

    if any(is_bi_i) && any(~is_bi_i)
        [b_i,~,stats_i] = glmfit(e, is_bi_i, 'binomial','logit');
        beta_all(i) = b_i(2);
        p_all(i)    = stats_i.p(2);
    end
end

figure('Color','w')
plot(beta_all, 'ko-','LineWidth',1.5)
xlabel('Offer index')
ylabel('Logistic slope (error → bistability)')
title('Relationship peaks at highest-conflict offer?')
%%
trial_range = 20;
window = [0 3];binwidth = 0.2;
subplot(2,1,1)
histogram(err_xhat_sorted(fp_stable_counts(1:trial_range,6)>1 &...
    fp_stable_counts(1:trial_range,9)>1),'binwidth',binwidth)
xlim(window)

subplot(2,1,2)
histogram(err_xhat_sorted(fp_stable_counts(1:trial_range,6)==1 &...
    fp_stable_counts(1:trial_range,9)==1),'binwidth',binwidth)
xlim(window)
%%
plot(smooth(fp_stable_counts(:,9),50))
hold on
plot(smooth(fp_x0_stable_counts(:,9),50))
xlim([0 700])
%%
win = 50;

y1 = smooth(fp_stable_counts(1:700,9)>=2, win);
y2 = smooth(fp_x0_stable_counts(1:700,9)>=2, win);

figure('Color','w'); hold on

% lines
h1 = plot(y1, '-', 'LineWidth', 2);
h2 = plot(y2, '-', 'LineWidth', 2);

% labels
xlabel('Fit rank (low → high error)')
ylabel(sprintf('Stable fixed points (smoothed, window=%d)', win))
title('Bistability across fits (offer 9)')

% legend
legend([h1 h2], {'Optimized (xhat)', 'Initial (x0)'}, ...
    'Location','best', 'Box','off')

% axes styling
set(gca, ...
    'TickDir','out', ...
    'Box','off', ...
    'LineWidth',1.2, ...
    'FontSize',12)

% tighten layout
xlim([1 600])

% optional: make figure proportions nicer for papers
set(gcf,'Position',[100 100 500 350])
%%
plot(smooth(fp_stable_counts(:,11),20))
hold on
plot(smooth(fp_x0_stable_counts(:,11),20))
%%
err_xhat = arrayfun(@(s) s.err, fit1.allfits);

[err_sorted, sort_idx] = sort(err_xhat, 'ascend');
%[delta_sorted, sort_idx] = sort(delta_err, 'ascend');

is_bistable = fp_stable_counts > 1;

is_bistable_sorted = is_bistable(sort_idx, :);

nfit1s = size(is_bistable_sorted,1);


low_idx  = 1:500;
high_idx = 50:100;

p_bistable_low  = mean(is_bistable_sorted(low_idx, :), 1);
p_bistable_high = mean(is_bistable_sorted(high_idx, :), 1);

conflict = target.conf_mean(:)';   % (1 × nOffers)

[r_low,  p_low]  = corr(p_bistable_low(:),  conflict(:), 'type', 'Spearman');
[r_high, p_high] = corr(p_bistable_high(:), conflict(:), 'type', 'Spearman');

figure; hold on

% Scatter
plot(conflict, p_bistable_low,  'bo', 'MarkerSize',8, 'LineWidth',1.5)
plot(conflict, p_bistable_high, 'ro', 'MarkerSize',8, 'LineWidth',1.5)

% ----- Best fit1 lines (linear fit1 for visualization) -----
xfit1 = linspace(min(conflict), max(conflict), 100);

% LOW error fit1 line
coef_low = polyfit(conflict, p_bistable_low, 1);
yfit1_low = polyval(coef_low, xfit1);
plot(xfit1, yfit1_low, 'b-', 'LineWidth',2)

% HIGH error fit1 line
coef_high = polyfit(conflict, p_bistable_high, 1);
yfit1_high = polyval(coef_high, xfit1);
plot(xfit1, yfit1_high, 'r-', 'LineWidth',2)

% Labels
xlabel('Conflict')
ylabel('P(bistable)')
legend({'Low error fit1s','High error fit1s'}, 'Location','best')
title('Bistability vs Conflict')

% ----- Annotate stats -----
txt_low = sprintf('Low err: \\rho = %.2f, p = %.3g', r_low, p_low);
txt_high = sprintf('High err: \\rho = %.2f, p = %.3g', r_high, p_high);

% place text nicely
xpos = min(conflict) + 0.05*range(conflict);
ypos = max([p_bistable_low p_bistable_high]);

text(xpos, ypos, txt_low,  'Color','b', 'FontSize',11, 'VerticalAlignment','top')
text(xpos, ypos-0.08, txt_high, 'Color','r', 'FontSize',11, 'VerticalAlignment','top')

axis square
box on

%%
err_xhat = arrayfun(@(s) s.err, fit1.allfits);
[err_sorted, sort_idx] = sort(err_xhat, 'ascend');

is_bistable = fp_stable_counts > 1;
is_bistable_sorted = is_bistable(sort_idx, :);

conflict = target.conf_mean(:)';

nfit1s = size(is_bistable_sorted,1);

bin_size = 50;
nBins = 5;

% ----- Color gradient: hot pink -> blue -----
c_start = [1 0 0.5];   % hot pink
c_end   = [0 0 1];     % blue

colors = zeros(nBins,3);
for i = 1:nBins
    t = (i-1)/(nBins-1);
    colors(i,:) = (1-t)*c_start + t*c_end;
end

figure; hold on

xfit1 = linspace(min(conflict), max(conflict), 100);

for b = 1:nBins

    idx_start = (b-1)*bin_size + 1;
    idx_end   = b*bin_size;

    idx = idx_start:idx_end;

    % Mean bistability for this bin
    p_bistable = mean(is_bistable_sorted(idx,:), 1);

    % Linear fit1
    coef = polyfit(conflict, p_bistable, 1);
    yfit1 = polyval(coef, xfit1);

    % Plot line only
    plot(xfit1, yfit1, 'Color', colors(b,:), 'LineWidth', 2)

end

xlabel('Conflict')
ylabel('P(bistable)')
title('Bistability vs Conflict across fit1 quality (gradient)')

axis square
box on