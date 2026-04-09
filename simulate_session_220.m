function S = simulate_session_220(params, offers, post, sessioncfg)

    nTypes   = numel(offers.reward);
    nTrials  = sessioncfg.n_trials_session;
    burnin   = sessioncfg.burnin;
    smooth_n = sessioncfg.smooth_post;

    % ---------------------------------
    % Make a 220-trial session layout
    % approximately balanced across trial types
    % ---------------------------------
    base_reps = floor(nTrials / nTypes);
    remainder = mod(nTrials, nTypes);

    trial_order = repelem(1:nTypes, base_reps);
    if remainder > 0
        extra_types = randperm(nTypes, remainder);
        trial_order = [trial_order, extra_types];
    end
    trial_order = trial_order(randperm(numel(trial_order)));

    % ---------------------------------
    % Preallocate session-level outputs
    % ---------------------------------
    S.trial_order   = trial_order(:);
    S.reward_trial  = offers.reward(trial_order(:))';
    S.punish_trial  = offers.punish(trial_order(:))';

    S.decision        = nan(nTrials,1);
    S.decision_time   = nan(nTrials,1);   % relative to post-burnin segment
    S.num_states      = nan(nTrials,1);
    S.num_pos_states  = nan(nTrials,1);
    S.num_neg_states  = nan(nTrials,1);
    S.conflict_trial  = nan(nTrials,1);

    % full trajectories for plotting from true trial onset
    S.trajectory_x = cell(nTrials,1);
    S.trajectory_y = cell(nTrials,1);

    % optional: also keep the post-burnin segment explicitly
    S.trajectory_x_postburn = cell(nTrials,1);
    S.trajectory_y_postburn = cell(nTrials,1);

    S.posterior    = cell(nTrials,1);
    S.entry_times  = cell(nTrials,1);

    % ---------------------------------
    % Simulate each session trial
    % ---------------------------------
    for tr = 1:nTrials
        tt = trial_order(tr);
        reward     = offers.reward(tt);
        punishment = offers.punish(tt);

        [trajx_full, trajy_full] = mfsim_fitting_plot_nullclines_fast( ...
            reward, punishment, 0, 0, ...
            params.alphareward, params.alphapunish, params.noise, ...
            params.lambda, params.w_i, params.a, ...
            params.b, params.c, 1);

        % keep full trajectories for plotting
        S.trajectory_x{tr} = trajx_full;
        S.trajectory_y{tr} = trajy_full;

        % make post-burnin copy for state detection
        if burnin > 0 && numel(trajx_full) > burnin
            trajx_post = trajx_full(burnin+1:end);
            trajy_post = trajy_full(burnin+1:end);
        else
            trajx_post = trajx_full;
            trajy_post = trajy_full;
        end

        S.trajectory_x_postburn{tr} = trajx_post;
        S.trajectory_y_postburn{tr} = trajy_post;

        posteriors = compute_posterior_timeseries(trajx_post, trajy_post, post);

        if smooth_n > 1
            posteriors = smooth(posteriors, smooth_n);
        end

        [decision_time, decision, num_state, num_pos, num_neg, entry_times] = ...
            detectStatePreferences(posteriors, params.time_stable, params.thresh);

        conflict_val = mean(1 - abs(2*posteriors - 1), 'omitnan');

        S.decision(tr)       = decision;
        S.decision_time(tr)  = decision_time;
        S.num_states(tr)     = num_state;
        S.num_pos_states(tr) = num_pos;
        S.num_neg_states(tr) = num_neg;
        S.conflict_trial(tr) = conflict_val;

        S.posterior{tr}   = posteriors;
        S.entry_times{tr} = entry_times;
    end

    % ---------------------------------
    % Summarize by trial type
    % ---------------------------------
    S.trial_type = struct();

    S.p_approach          = nan(1,nTypes);
    S.conflict            = nan(1,nTypes);
    S.avg_decision_time   = nan(1,nTypes);
    S.avg_num_states      = nan(1,nTypes);
    S.avg_transition_rate = nan(1,nTypes);

    for i = 1:nTypes
        idx = find(S.trial_order == i);

        decisions_i      = S.decision(idx);
        decision_times_i = S.decision_time(idx);
        num_states_i     = S.num_states(idx);
        num_pos_i        = S.num_pos_states(idx);
        num_neg_i        = S.num_neg_states(idx);
        conflict_i       = S.conflict_trial(idx);

        S.trial_type(i).trial_indices   = idx;
        S.trial_type(i).decisions       = decisions_i;
        S.trial_type(i).decision_times  = decision_times_i;
        S.trial_type(i).num_states      = num_states_i;
        S.trial_type(i).num_pos_states  = num_pos_i;
        S.trial_type(i).num_neg_states  = num_neg_i;
        S.trial_type(i).conflict_trial  = conflict_i;

        S.p_approach(i) = mean(decisions_i == 1, 'omitnan');

        prob = S.p_approach(i);
        if isnan(prob) || prob == 0 || prob == 1
            S.conflict(i) = 0;
        else
            S.conflict(i) = -prob*log2(prob) - (1-prob)*log2(1-prob);
        end

        S.avg_decision_time(i)   = mean(decision_times_i, 'omitnan');
        S.avg_num_states(i)      = mean(num_states_i, 'omitnan');
        S.avg_transition_rate(i) = mean(num_states_i, 'omitnan');
    end
end