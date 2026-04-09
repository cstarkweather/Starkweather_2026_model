function p = compute_posterior_timeseries(trajx, trajy, post)
% Returns p(approach) for each timepoint.

trajx = trajx(:);
trajy = trajy(:);

pxA = normpdf(trajx, post.mu_x_appr, post.sd_x_appr);
pyA = normpdf(trajy, post.mu_y_appr, post.sd_y_appr);
pA  = pxA .* pyA;

pxB = normpdf(trajx, post.mu_x_av, post.sd_x_av);
pyB = normpdf(trajy, post.mu_y_av, post.sd_y_av);
pB  = pxB .* pyB;

num = pA .* post.pi_appr;
den = num + pB .* post.pi_av + eps;

p = num ./ den;
end