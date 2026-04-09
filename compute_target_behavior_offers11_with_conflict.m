function target = compute_target_behavior_offers11_with_conflict(behavior_data, offers11)

nSubj  = numel(behavior_data);
nTypes = size(offers11,1);

pA = nan(nSubj, nTypes);
rt = nan(nSubj, nTypes);
cf = nan(nSubj, nTypes);

for s = 1:nSubj
    D  = behavior_data{s}.decision(:);        % 0/1
    RT = behavior_data{s}.reaction_time(:);   % ms (looks like ms in your target)
    R  = behavior_data{s}.reward_trial(:);
    P  = behavior_data{s}.punish_trial(:);

    ok = isfinite(D) & isfinite(RT) & isfinite(R) & isfinite(P);
    D = D(ok); RT = RT(ok); R = R(ok); P = P(ok);

    for t = 1:nTypes
        r0 = offers11(t,1);
        p0 = offers11(t,2);
        idx = (R==r0) & (P==p0);

        if any(idx)
            pA(s,t) = mean(D(idx)==1);
            rt(s,t) = mean(RT(idx));                 % ms
            prob = pA(s,t);
            if prob==0 || prob==1
                cf(s,t) = 0;
            else
                cf(s,t) = -prob*log2(prob) - (1-prob)*log2(1-prob);
            end
        end
    end
end

target.offers11 = offers11;
target.trialIDs = 1:nTypes;

target.pA_mean = mean(pA,1,'omitnan');
target.pA_se   = std(pA,0,1,'omitnan') ./ sqrt(sum(isfinite(pA),1));

target.rt_mean = mean(rt,1,'omitnan');
target.rt_se   = std(rt,0,1,'omitnan') ./ sqrt(sum(isfinite(rt),1));

target.conf_mean = mean(cf,1,'omitnan');
target.conf_se   = std(cf,0,1,'omitnan') ./ sqrt(sum(isfinite(cf),1));

target.pA_bySubj = pA;
target.rt_bySubj = rt;
target.conf_bySubj = cf;

target.nSubj_per_type = sum(isfinite(pA),1);
end