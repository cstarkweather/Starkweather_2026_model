function [decision_time, decision, num_states, num_pos_entries, num_neg_entries, entry_times] = ...
    detectStatePreferences(p, time_stable, thresh)

    % detectStatePreferences with:
    %  - minimum decision time of 600 samples
    %  - minimum dwell time of 5 samples for counting a state

    % ---------- sanitize input ----------
    p = p(:);
    T = numel(p);

    if T == 0
        decision_time = 0;
        decision = NaN;
        num_states = 0;
        num_pos_entries = 0;
        num_neg_entries = 0;
        entry_times = [];
        return
    end

    time_stable = round(time_stable);
    min_decision_time = 600;
    min_dwell = 5;   % 5 ms if sampled at 1 kHz

    % fill NaNs as neutral evidence
    p_filled = p;
    p_filled(isnan(p_filled)) = 0.5;

    % ---------- smooth posterior ----------
    p_smooth = p_filled;

    % ---------- convert to binary state ----------
    isPos_raw = p_smooth > 0.5;

    % ---------- remove brief state flips (< min_dwell) ----------
    isPos = enforce_min_dwell(isPos_raw, min_dwell);

    % ---------- fallback values ----------
    decision_time = T;
    decision = round(p_smooth(end));

    % ---------- detect first stable window ----------
    isAbove = double(isPos);

    if T >= time_stable
        winSum  = conv(isAbove, ones(time_stable,1), 'valid');
        winFrac = winSum / time_stable;

        stablePos = (winFrac >= thresh);
        stableNeg = (winFrac <= (1 - thresh));
        stableAny = stablePos | stableNeg;

        candidate_start = find(stableAny, 1, 'first');

        if ~isempty(candidate_start)
            candidate_end = candidate_start + time_stable - 1;

            if candidate_end <= min_decision_time
                % stability already achieved by 600 ms
                decision_time = min_decision_time;

                eval_start = max(1, min_decision_time - time_stable + 1);
                eval_end   = eval_start + time_stable - 1;

                frac_pos = mean(isPos(eval_start:eval_end));
                if frac_pos >= thresh
                    decision = 1;
                elseif frac_pos <= (1 - thresh)
                    decision = 0;
                else
                    decision = mean(isPos(eval_start:eval_end)) > 0.5;
                end

            else
                valid_starts = find( ...
                    stableAny & ((1:numel(stableAny))' + time_stable - 1 >= min_decision_time), ...
                    1, 'first');

                if ~isempty(valid_starts)
                    decision_time = valid_starts + time_stable - 1;
                    decision = stablePos(valid_starts);
                end
            end
        end
    end

    % ---------- compute state segmentation up to decision_time ----------
    full_state = isPos(1:decision_time);

    segStart = [true; diff(full_state) ~= 0];
    segIdx   = find(segStart);

    % segment ends
    segEnd = [segIdx(2:end)-1; numel(full_state)];
    segLen = segEnd - segIdx + 1;
    segSign = full_state(segIdx);

    % keep only segments that last at least min_dwell
    keep = segLen >= min_dwell;

    entry_times = segIdx(keep);
    segSign_kept = segSign(keep);

    num_states = numel(segSign_kept);
    num_pos_entries = sum(segSign_kept == 1);
    num_neg_entries = sum(segSign_kept == 0);
end


function state_out = enforce_min_dwell(state_in, min_dwell)
    % Replace short runs (< min_dwell) with the previous state if possible,
    % otherwise with the next state.

    state_in = logical(state_in(:));
    N = numel(state_in);

    if N == 0
        state_out = state_in;
        return
    end

    state_out = state_in;

    changed = true;
    while changed
        changed = false;

        runStart = [1; find(diff(state_out) ~= 0) + 1];
        runEnd   = [runStart(2:end)-1; N];
        runLen   = runEnd - runStart + 1;
        runVal   = state_out(runStart);

        shortRuns = find(runLen < min_dwell);

        if isempty(shortRuns)
            break
        end

        for i = 1:numel(shortRuns)
            r = shortRuns(i);
            s = runStart(r);
            e = runEnd(r);

            if r > 1
                fillVal = runVal(r-1);
            elseif r < numel(runVal)
                fillVal = runVal(r+1);
            else
                fillVal = runVal(r);
            end

            state_out(s:e) = fillVal;
            changed = true;
        end
    end
end