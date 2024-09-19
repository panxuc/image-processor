function category = dc_error_to_category(dc_error)
    category = ceil(log2(abs(dc_error) + 1));
end
