clear store_rho store_p store_sig

% fp_stable_counts is already sorted by ascending xhat error
% errs_sorted is the matching sorted error vector
e_all = errs_sorted(:);

% optional: keep same valid subset you used before
valid = e_all < max(e_all(1:793));

% already-sorted bistability matrix
is_bistable_all = fp_stable_counts > 1;

% apply same valid mask to both
e_use    = e_all(valid);
is_bi_use = is_bistable_all(valid,:);

conflict = target.conf_mean(:);   % nOffers x 1
nFits_use = size(is_bi_use, 1);

store_rho = nan(nFits_use,1);
store_p   = nan(nFits_use,1);
store_sig = false(nFits_use,1);

for iFit = 1:nFits_use
    y = double(is_bi_use(iFit,:))';   % 0/1 across offers

    % skip degenerate fits
    if numel(unique(y)) < 2
        continue
    end

    [rho, pval] = corr(conflict, y, 'Type', 'Spearman', 'Rows', 'complete');

    store_rho(iFit) = rho;
    store_p(iFit)   = pval;
    store_sig(iFit) = pval < 0.05;
end

%%
win = 50;

rho = store_rho(:);
x = (1:numel(rho))';

% trailing moving mean
y = movmean(rho, [win-1 0], 'omitnan');

% trailing moving SEM
sem = nan(size(rho));
for i = 1:numel(rho)
    i0 = max(1, i-win+1);
    vals = rho(i0:i);
    vals = vals(~isnan(vals));

    if numel(vals) >= 2
        sem(i) = std(vals) / sqrt(numel(vals));
    end
end

upper = y + 2*sem;
lower = y - 2*sem;

figure('Color','w'); hold on

% error patch
good = ~isnan(x) & ~isnan(lower) & ~isnan(upper);
patch([x(good); flipud(x(good))], ...
      [upper(good); flipud(lower(good))], ...
      'k', ...
      'FaceAlpha', 0.15, ...
      'EdgeColor', 'none')

% raw points
plot(x, rho, 'o', ...
    'MarkerSize', 4, ...
    'MarkerFaceColor', [0.75 0.75 0.75], ...
    'MarkerEdgeColor', 'none')

% smoothed line
plot(x, y, 'k-', 'LineWidth', 2)

% zero line
yline(0, '--k', 'LineWidth', 1)

xlabel('Fit rank (low \rightarrow high error)')
ylabel(sprintf('Spearman \\rho (trailing mean \\pm 2 SEM, window = %d)', win))
title('Conflict–bistability relationship across fits')

xlim([1 600])
ylim([-1 1])

set(gca, ...
    'TickDir','out', ...
    'Box','off', ...
    'LineWidth',1.2, ...
    'FontSize',12)

set(gcf,'Position',[100 100 500 350])
%%
clear store_rho store_p store_sig
clear store_rho_x0 store_p_x0 store_sig_x0

% ------------------------------------------------------------
% OPTIMIZED FITS (xhat)
% assumes:
%   errs_sorted            = sorted fit errors for xhat fits
%   fp_stable_counts       = matching stable fixed-point counts, sorted
%   target.conf_mean       = conflict per offer
% ------------------------------------------------------------
e_all = errs_sorted(:);

% keep same valid subset you used before
valid = e_all < max(e_all(1:end));

% bistability matrix for optimized fits
is_bistable_all = fp_stable_counts > 1;

% apply mask
e_use     = e_all(valid);
is_bi_use = is_bistable_all(valid,:);

conflict = target.conf_mean(:);   % nOffers x 1
nFits_use = size(is_bi_use, 1);

store_rho = nan(nFits_use,1);
store_p   = nan(nFits_use,1);
store_sig = false(nFits_use,1);

for iFit = 1:nFits_use
    y = double(is_bi_use(iFit,:))';   % 0/1 across offers

    if numel(unique(y)) < 2
        continue
    end

    [rho, pval] = corr(conflict, y, 'Type', 'Spearman', 'Rows', 'complete');

    store_rho(iFit) = rho;
    store_p(iFit)   = pval;
    store_sig(iFit) = pval < 0.05;
end

% ------------------------------------------------------------
% PRE-OPTIMIZATION STARTS (x0)
% assumes:
%   fp_x0_stable_counts    = stable fixed-point counts for x0 fits
%   fp_x0_counts / info    = already aligned to the same fits as allfits
% if you have x0 errors separately, use them. otherwise just use all x0s.
% ------------------------------------------------------------
fp_x0_stable_counts = fp_x0_stable_counts(idx,:);
is_bistable_x0_all = fp_x0_stable_counts > 1;

% if x0 rows correspond 1:1 with the same sorted fit order:
is_bistable_x0_use = is_bistable_x0_all(valid,:);

nFits_x0_use = size(is_bistable_x0_use, 1);

store_rho_x0 = nan(nFits_x0_use,1);
store_p_x0   = nan(nFits_x0_use,1);
store_sig_x0 = false(nFits_x0_use,1);

for iFit = 1:nFits_x0_use
    y = double(is_bistable_x0_use(iFit,:))';

    if numel(unique(y)) < 2
        continue
    end

    [rho, pval] = corr(conflict, y, 'Type', 'Spearman', 'Rows', 'complete');

    store_rho_x0(iFit) = rho;
    store_p_x0(iFit)   = pval;
    store_sig_x0(iFit) = pval < 0.05;
end

% ------------------------------------------------------------
% HISTOGRAMS
% ------------------------------------------------------------
figure;
histogram(store_rho, 'BinMethod', 'fd');
xlabel('Spearman \rho');
ylabel('Count');
title('Optimized fits: correlation of conflict with bistability');
xline(0, '--k');

figure;
histogram(store_rho_x0, 'BinMethod', 'fd');
xlabel('Spearman \rho');
ylabel('Count');
title('Pre-optimization starts (x0): correlation of conflict with bistability');
xline(0, '--k');
