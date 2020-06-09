
function tmp(code)
    if code == 1
        CSV.read(datadep"medium_main.csv")
    elseif code == 2
        CSV.read(datadep"medium_controls_main.csv")
    elseif code == 3
        CSV.read(datadep"full_main.csv")
    elseif code == 4
        CSV.read(datadep"full_controls_main.csv")
    end
end
