function params = unpack_dyn(x)
    params.alphareward = x(1);
    params.alphapunish = x(2);
    params.noise       = x(3);
    params.w_i         = x(4);
    params.a           = x(5);
    params.lambda      = x(6);
    params.b           = x(7);
    params.c           = x(8);
    params.time_stable = round(x(9));
    params.thresh      = x(10);
end