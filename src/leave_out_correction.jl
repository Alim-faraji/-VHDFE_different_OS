
## Dependencies

#To install Laplacians you need to put in Julia's Prompt "add Laplacians#master"
#For some reason Pkg.add() doesn't work with this.
#Pkg.add("MATLAB")
#using Laplacians
#using Pkg
#include(string(Pkg.dir("Laplacians") , "/src/matlabSolvers.jl"))
getlagged(x) = [NaN; x[1:(end - 1)]]

## Exported Methods and Types

export find_connected_set,prunning_connected_set,drop_single_obs, index_constr, eff_res


#Defining types and structures
abstract type AbstractLLSAlgorithm end
abstract type AbstractGraphLLSAlgorithm <: AbstractLLSAlgorithm end  # i.e. requires graph laplacian
struct CMGPreconditionedLLS <: AbstractGraphLLSAlgorithm  end
struct AMGPreconditionedLLS <: AbstractGraphLLSAlgorithm  end
struct DirectLLS <: AbstractLLSAlgorithm  end  # no graph required

abstract type AbstractLeverageAlgorithm end
struct ExactAlgorithm <: AbstractLeverageAlgorithm  end
@with_kw struct JLAAlgorithm <: AbstractLeverageAlgorithm  
    num_simulations::Int64 = 0
end

@with_kw struct Settings{LLSAlgorithm, LeverageAlgorithm}
    cg_maxiter::Int64 = 300
    lls_algorithm::LLSAlgorithm = AMGPreconditionedLLS()
    leverage_algorithm::LeverageAlgorithm = ExactAlgorithm()
    clustering_level::String = "obs"
    person_effects::Bool = false
    cov_effects::Bool = false
end


## Methods and Types Definitions


#1) Finds AKM largest connected set
function find_connected_set(y, idvar, firmidvar; verbose=false)

    #Observation identifier to join later the FE 
    obs_id = collect(1:size(y,1))

    firms = unique(firmidvar)
    firmid = indexin(firmidvar, firms)
    workers = unique(idvar)
    id = indexin(idvar, workers)
    
    #Save ids just in case
    firmid_old=firmidvar;
    id_old=idvar;
    
    n_firms=length(firms)
    n_workers=length(workers)
    firmid=firmid.+n_workers
    
    graph_size=length(firms)+length(workers)
    G=Graph(graph_size)
    for i in 1:size(y,1)
        add_edge!(G, firmid[i], id[i])
    end
    
    cs=connected_components(G)
    if verbose == true
        println("Largest Connected Set has been found")
    end
    pos = indexin(  [  maximum(size.(cs,1))] , size.(cs,1) )[1]
    lcs=cs[pos]

    connected_firms=lcs[lcs.>n_workers]
    
    sel=findall(in(connected_firms),firmid)

    obs_id = [obs_id[x] for x in sel ]
    yvec = [y[x] for x in sel ]
    firmid = [firmidvar[x] for x in sel ]
    id = [idvar[x] for x in sel ]

    #Relabel firms
    firms = unique(firmid)
    firmid = indexin(firmid, firms)

    #Relabel workers
    ids = unique(id)
    id = indexin(id, ids)  

    return (obs_id = obs_id , y = yvec , id = id, firmid = firmid )

end


#2) Pruning and finding Leave-Out Largest connected set
function prunning_connected_set(yvec, idvar, firmidvar, obs_id; verbose=false)

    firms = unique(firmidvar)
    firmid = indexin(firmidvar, firms)
    workers = unique(idvar)
    id = indexin(idvar, workers)

    nbadworkers=1
    while nbadworkers>0

        n_firms=length(firms)
        n_workers=length(workers)
        firmid=firmid.+n_workers

        graph_size=n_workers + n_firms
        lcs_graph=Graph(graph_size)
            for i in 1:size(yvec,1)
                add_edge!(lcs_graph, firmid[i], id[i])
            end

        #get articulation vertex
        artic_vertex = articulation(lcs_graph)

        sel=findall(x->x==nothing, indexin(id,artic_vertex))
        
        bad_workers = artic_vertex[artic_vertex.<=n_workers]
        nbadworkers = size(bad_workers,1)

        if verbose==true
            println("Number of workers that are articulation points: ", nbadworkers)
        end

        #Restrict the sample 
        yvec = [yvec[x] for x in sel ]
        firmid = [firmid[x] for x in sel ]
        id = [id[x] for x in sel ]
        obs_id = [obs_id[x] for x in sel ]

        #Relabel firms
        firms = unique(firmid)
        firmid = indexin(firmid, firms)

        #Relabel workers
        ids = unique(id)
        id = indexin(id, ids)

        n_workers=maximum(id)
        n_firms=maximum(firmid);

        firmid = firmid .+ n_workers

        #Constructing new Graph
        G=Graph(n_workers+n_firms)
        Nprimeprime = size(sel,1)
        for i in 1:Nprimeprime
            add_edge!(G, firmid[i], id[i])
        end

        #Find Largest Connected Set (LCS)
        cs=connected_components(G)

        pos = indexin(  [  maximum(size.(cs,1))] , size.(cs,1) )[1]
        lcs=cs[pos]

        connected_firms=lcs[lcs.>n_workers]


        sel=findall(in(connected_firms),firmid)

        obs_id = [obs_id[x] for x in sel ]
        yvec = [yvec[x] for x in sel ]
        firmid = [firmid[x] for x in sel ]
        id = [id[x] for x in sel ]

        #Relabel firms
        firms = unique(firmid)
        firmid = indexin(firmid, firms)

        #Relabel workers
        ids = unique(id)
        id = indexin(id, ids)

    end

    

    return (obs_id = obs_id , y = yvec , id = id, firmid = firmid )
end

#3) Drops Single Observations 

function drop_single_obs(yvec, idvar, firmidvar,obs_id)

    firms = unique(firmidvar)
    firmid = indexin(firmidvar, firms)
    workers = unique(idvar)
    id = indexin(idvar, workers)  

    T = Matlab.accumarray(id,1)
    T = [T[x] for x in id]
    sel = T.>1
    sel = findall(x->x==true,sel )

    obs_id = [obs_id[x] for x in sel ]
    yvec = [yvec[x] for x in sel ]
    firmid = [firmid[x] for x in sel ]
    id = [id[x] for x in sel ]

    #Relabel firms
    firms = unique(firmid)
    firmid = indexin(firmid, firms)

    #Relabel workers
    ids = unique(id)
    id = indexin(id, ids)
    
    return (obs_id = obs_id , y = yvec , id = id, firmid = firmid )
end



#4) Finds the observations connected for every cluster

function index_constr(clustering_var, id, match_id )
    NT = length(clustering_var)
    counter = ones(size(clustering_var,1));
    
    #Indexing obs number per worked/id
    gcs = Int.(@transform(groupby(DataFrame(counter = counter, id = id), :id), gcs = cumsum(:counter)).gcs);
    maxD = maximum(gcs);

    index = collect(1:NT);
    
    #This will be the output, and we will append observations to it in the loop
    list_final=DataFrame(row = Int64[],col = Int64[], match_id = Int64[], id_cluster = Int64[]);


    for t=1:maxD
    rowsel =  findall(x->x==true,gcs.==t);
    list_base = DataFrame( id_cluster= [clustering_var[x] for x in rowsel], row = 
    [index[x] for x in rowsel] , match_id = [match_id[x] for x in rowsel]  );

        for tt=t:maxD
            colsel =  findall(x->x==true,gcs.==tt);
            list_sel =  DataFrame( id_cluster= [clustering_var[x] for x in colsel], col = 
            [index[x] for x in colsel] );

            merge = outerjoin(list_base, list_sel, on = :id_cluster)
            merge = dropmissing(merge)
            merge = merge[:,[:row, :col, :match_id, :id_cluster]]

            append!(list_final, merge)

        end
    end

    sort!(list_final, (:row));

    return Matrix(list_final)

end

function compute_movers(id,firmid)

    gcs = [NaN; id[1:end-1]]
    gcs = id.!=gcs 

    lagfirmid=[NaN; firmid[1:end-1]]
    for x in findall(x->x ==true , gcs) 
        lagfirmid[x] = NaN 
    end

    stayer=(firmid.==lagfirmid)
    for x in findall(x->x ==true , gcs) 
        stayer[x] = true 
    end

    stayer=Int.(stayer)
    stayer=Matlab.accumarray(id,stayer)
    T=Matlab.accumarray(id,1)
    stayer=T.==stayer
    movers=stayer.==false

    movers = [movers[x] for x in id]
    T = [T[x] for x in id]


    return (movers = movers, T = T)
end

##Least Squares Problem 

function lss(::AMGPreconditionedLLS, X,y, settings)
    #Solves using cg
    xx = SparseMatrixCSC{Float64,Int64}(X'*X)
    P = aspreconditioner(ruge_stuben(xx))

    sol = cg!(0.1.* ones(length(y)), xx, SparseMatrixCSC{Float64,Int64}(y), Pl = P , log=true, maxiter=settings.cg_maxiter)

    return sol[1]
end


function lss(::DirectLLS, X,y, settings)
    #Solves using cg
    xx = Matrix(X'*X)     
    return   xx \Array(y) 
end


#function lss(::CMGPreconditionedLLS, X,y, settings)
    #Solves using cg
#    xx = SparseMatrixCSC{Float64,Int64}(X'*X)
#    yprime = Array{Float64,1}(y)
#    sol=matlabCmgSolver(xx, yprime; tol=1e-6, maxits=settings.cg_maxiter);
#    return   sol
#end



##Computes Pii and Bii


function compute_leverages(::ExactAlgorithm, X,y, M, NT, N, J ,settings; scale=250)
 
    zexact_i = lss(settings.lls_algorithm, X, y, settings)    

    #Compute Pii 
    Pii =y'*zexact_i

    #Compute Bii firm effects
    aux_right = zexact_i[N+1:N+J-1,:]
    aux_left= zexact_i[N+1:N+J-1,:]

    COV=cov(X[:,N+1:N+J-1]*aux_left,X[:,N+1:N+J-1]*aux_right)
    Bii_fe =COV[1]*(NT-1)

    #Compute Bii person effects
    if settings.person_effects==true
        aux_right=zexact_i[1:N]
        aux_left=zexact_i[1:N]
        COV=cov(X[:,1:N]*aux_left,X[:,1:N]*aux_right)
        Bii_pe=COV[1]*(NT-1)
    end

    #Compute Bii cov(pe,fe)
    if settings.cov_effects==true
        aux_right = zexact_i[N+1:N+J-1]
        aux_left=zexact_i[1:N]
        COV=cov(X[:,1:N]*aux_left,X[:,N+1:N+J-1]*aux_right)
        Bii_cov=COV[1]*(NT-1)
    end

    if settings.person_effects==false & settings.cov_effects==false
        return (Pii = Pii, Bii_fe=Bii_fe)
    elseif settings.person_effects==true & settings.cov_effects==false
        return (Pii = Pii, Bii_fe=Bii_fe, Bii_pe = Bii_pe)
    elseif settings.person_effects==false & settings.cov_effects==true
        return (Pii = Pii, Bii_fe=Bii_fe, Bii_cov = Bii_cov)
    elseif settings.person_effects==true & settings.cov_effects==true
        return (Pii = Pii, Bii_fe=Bii_fe, Bii_pe = Bii_pe, Bii_cov = Bii_cov)
    end
end


function compute_leverages(::JLAAlgorithm, X,AUX, M, NT, N, J , settings; scale=250)

    #This function returns vectors, not scalars. 

    #Actually scale/eps should be part of JLAA struct, remember to modify that
    #epsilon = 0.005
    #scale = ceil(log2(NT)/epsilon)

    #Rademacher Entries
    rademach = rand(1,NT) .> 0.5
    rademach = rademach - (rademach .== 0)
    rademach = rademach ./sqrt(scale)

    Z  = lss(settings.lls_algorithm, X, (rademach * X), settings)

    rademach = rademach .- mean(rademach)
    ZB = lss(settings.lls_algorithm, X, (rademach * AUX.Xfe), settings)

    if settings.person_effects == true 
        ZB_pe = lss(settings.lss_algorithm, X, (rademach * AUX.Xpe), settings)
    end    

    #Computing
    Pii = ( [Z[j]  for j in AUX.elist_1 ]  .- [Z[j]  for j in AUX.elist_2 ] ) .* ( [Z[j]  for j in AUX.elist_3 ]  .- [Z[j]  for j in AUX.elist_4 ] )
    Bii_fe = ( [ZB[j]  for j in AUX.elist_1 ]  .- [ZB[j]  for j in AUX.elist_2 ] ) .* ( [ZB[j]  for j in AUX.elist_3 ]  .- [ZB[j]  for j in AUX.elist_4 ] )

    if settings.person_effects == true 
        Bii_pe =  ( [ZB_pe[j]  for j in AUX.elist_1 ]  .- [ZB_pe[j]  for j in AUX.elist_2 ] ) .* ( [ZB_pe[j]  for j in AUX.elist_3 ]  .- [ZB_pe[j]  for j in AUX.elist_4 ] )
    end    
    
    if settings.cov_effects == true 
        Bii_cov = ( [ZB[j]  for j in AUX.elist_1 ]  .- [ZB[j]  for j in AUX.elist_2 ] ) .* ( [ZB_pe[j]  for j in AUX.elist_3 ]  .- [ZB_pe[j]  for j in AUX.elist_4 ] )
    end

    #Output
    if settings.person_effects==false & settings.cov_effects==false
        return (Pii = Pii, Bii_fe=Bii_fe)
    elseif settings.person_effects==true & settings.cov_effects==false
        return (Pii = Pii, Bii_fe=Bii_fe, Bii_pe = Bii_pe)
    elseif settings.person_effects==false & settings.cov_effects==true
        return (Pii = Pii, Bii_fe=Bii_fe, Bii_cov = Bii_cov)
    elseif settings.person_effects==true & settings.cov_effects==true
        return (Pii = Pii, Bii_fe=Bii_fe, Bii_pe = Bii_pe, Bii_cov = Bii_cov)
    end
end


##Initialize dependent variable in algorithm

function initialize_auxiliary_variables(::ExactAlgorithm, X, elist, M, NT, N, J, K, settings)
    Xright = nothing 
    Xleft = nothing 

    if K> 0
        if settings.clustering_level == "obs"
            Xright=  X[elist[:,2],:] 
        elseif settings.clustering_level == "match" 
            Xright=  X[elist[:,2],:] 
            Xleft=  X[elist[:,1],:] 
        end

    elseif K == 0

        Xright = sparse((1:M),elist[:,1],1.0,M,N+J)
        Xright = Xright .+ sparse((1:M),elist[:,2],-1.0,M,N+J) 

    end

    return (Xright = Xright, Xleft = Xleft)

end

function initialize_auxiliary_variables(::JLAAlgorithm, X, elist, M,NT, N, J, K,  settings)
    if K ==0
        Xfe= hcat(spzeros(NT,N), X[:,N+1:end])
        Xpe=hcat(X[:,N], spzeros(NT,J))

        elist_1= elist[:,1]
        elist_2= elist[:,2]        
        elist_3= elist[:,3]
        elist_4= elist[:,4]
    elseif K>0 
        #I'm not sure!
    end 
    #scale = settings.scale 
    return  (Xfe = Xfe , Xpe = Xpe , elist_1 = elist_1 , elist_2 = elist_2 , elist_3 = elist_3 , elist_4 = elist_4)
end


## Eff res 
function eff_res(::ExactAlgorithm, X,id,firmid,match_id, K, settings)

    #Indexing Observations
    if settings.clustering_level == "obs"
        elist = index_constr(collect(1:length(id)), id, match_id )
    elseif settings.clustering_level == "match"
        elist = index_constr(match_id, id, match_id)
    else 
        error("Clustering level not possible to implement. \nOptions available are obs or match.")
    end
    
    #Dimensions
    NT=size(X,1)
    M=size(elist,1)
    J = maximum(firmid)
    N = maximum(id)
        
    #Initialize output
    Pii=zeros(M)
    Bii_fe=zeros(M)
    Bii_cov= settings.cov_effects ==true ? zeros(M,1) : nothing
    Bii_pe= settings.person_effects == true ? zeros(M,1) : nothing

    #No controls case
    if K == 0

        #Compute Auxiliaries
        movers , T = compute_movers(id, firmid)

        Nmatches = maximum(match_id)
        match_id_movers = [match_id[x] for x in findall(x->x==true, movers)]
        firmid_movers = [firmid[x] for x in findall(x->x==true, movers)]
        id_movers = [id[x] for x in findall(x->x==true, movers)]
        
        sel = unique(z -> match_id_movers[z], 1:length(match_id_movers))
        match_id_movers=match_id_movers[sel]
        firmid_movers=firmid_movers[sel]
        id_movers=id_movers[sel]
        
        maxT = maximum([T[x] for x in findall(x->x == false, movers)])
        
        counter = ones(Int,NT)
        gcs = Int.(@transform(groupby(DataFrame(counter = counter, id = id), :id), gcs = cumsum(:counter)).gcs)
        sel_stayers=(gcs.==1).*(movers.==false)
        stayers_matches_sel=[match_id[z] for z in findall(x->x == true , sel_stayers)]
        Tinv=1 ./T
        elist_JLL=[id_movers N.+firmid_movers id_movers N.+firmid_movers]

        M=size(elist_JLL,1)
        Pii_movers=zeros(M)
        Bii_fe_movers=zeros(M)
        Bii_cov_movers= settings.cov_effects ==true ? zeros(M) : nothing
        Bii_pe_movers= settings.person_effects == true ? zeros(M) : nothing

        #Initializing dependent variables for LSS
        AUX = initialize_auxiliary_variables(settings.leverage_algorithm, X, elist_JLL, M, NT, N, J, K, settings)

        for i=1:M

            OUT = compute_leverages(settings.leverage_algorithm, X, AUX.Xright[i,:], M, NT, N, J  , settings; scale= nothing)

            Pii_movers[i] = OUT.Pii 
            Bii_fe_movers[i] = OUT.Bii_fe

            if Bii_pe != nothing
                Bii_pe_movers[i] =OUT.Bii_pe
            end

            if Bii_cov != nothing
                Bii_cov_movers[i] =OUT.Bii_cov
            end

        end

        #Assign Step
        Pii_movers=sparse(match_id_movers,ones(Int,length(match_id_movers)),Pii_movers[:,1],Nmatches,1)
        Pii_stayers=sparse(stayers_matches_sel,ones(Int,length(stayers_matches_sel)),[Tinv[x] for x in findall(x->x==true,sel_stayers)],Nmatches,1)
        Pii=Pii_movers.+Pii_stayers

        Bii_fe=sparse(match_id_movers,ones(Int,length(match_id_movers)),Bii_fe_movers[:,1],Nmatches,1)

        if settings.cov_effects == true
            Bii_cov=sparse(match_id_movers,ones(Int,length(match_id_movers)),Bii_cov_movers[:,1],Nmatches,1)
        end

        if settings.person_effects == true
            Bii_pe=sparse(match_id_movers,ones(Int,length(match_id_movers)),Bii_pe_movers[:,1],Nmatches,1)
            stayers = .!movers

            #Threads.@threads for t=2:maxT #T=1 have Pii=1 so need to be dropped.
            for t=2:maxT
                sel=(gcs.==true).*stayers.*(T.==t)
                N_sel=sum(sel)
                if N_sel > 0
                    index_sel=findall(x->x==true,sel)
                    match_sel_aux=Int.([match_id[z] for z in index_sel])
                    first=index_sel[1]
                    Xuse=X[first,:]

                    ztilde =lss(settings.lls_algorithm, X, Xuse, settings)
                    
                    aux_right=ztilde[1:N]
                    aux_left=ztilde[1:N]

                    COV=cov(X[:,1:N]*aux_left,X[:,1:N]*aux_right)
                    Bii_pe_stayers=COV[1]*(NT-1)

                    Bii_pe_stayers=sparse(match_sel_aux,ones(Int,length(match_sel_aux)),Bii_pe_stayers,Nmatches,1)
                    Bii_pe=Bii_pe.+Bii_pe_stayers
                end
            end    
        end
   

   

    elseif K> 0 
        #AUX = initialize_auxiliary_variables(settings.lls_algorithm, X, elist, M,NT, N, J, K, settings)
        nparameters = N + J + K

        D=sparse(collect(1:NT),id,1)
        F=sparse(collect(1:NT),firmid,1)
        Dvar = hcat(  D, spzeros(NT,nparameters-N) )
        Fvar = hcat(spzeros(NT,N), F, spzeros(NT,nparameters-N-J) )
        #Wvar = hcat(spzeros(NT,N+J), controls ) 
        Xleft = X[elist[:,1],:]
        Xright = X[elist[:,2],:]

    Threads.@threads for i=1:M    
    #for i =1:2
            if settings.clustering_level == "obs"
                    Zleft = lss(settings.lls_algorithm, X, Xleft[i,:], settings)
                    #Zright = lss(settings.lls_algorithm, X, Xright, settings)

                    Pii = Pii .+ (X*Zleft).^2

                    aux = lss(settings.lls_algorithm, X, Fvar[i,:], settings)
                    ZF = X*aux 
                    
                    Bii_fe = Bii_fe .+ ZF.^2 ./ NT

                if settings.person_effects == true |    settings.cov_effects == true
                    aux = lss(settings.lls_algorithm, X, Dvar[i,:], settings)
                    ZD = X*aux
                end

                if settings.person_effects==true 
                    Bii_pe = Bii_pe .+ (ZD).^2 ./ NT
                end
                    
                if settings.cov_effects==true 
                    Bii_cov = Bii_cov .+ (ZD .* ZF ) ./ NT
                end
                    

            elseif settings.clustering_level == "match"

            end

        end

    end

    #Create matrices
    rows = elist[:,1]
    cols = elist[:,2]
    index_cluster = elist[:,3]

    if K==0
        Pii = [Pii[x] for x in index_cluster]
        Bii_fe = [Bii_fe[x] for x in index_cluster]

        println("Preparing to build Lambda matrices.\n")

        if settings.cov_effects == true
            Bii_cov = [Bii_cov[x] for x in index_cluster]
        end

        if settings.person_effects == true
            Bii_pe = [Bii_pe[x] for x in index_cluster]
        end

    end


    #Lambda P
    Lambda_P=sparse(rows,cols,Pii,NT,NT)
    Lambda_P=Lambda_P+triu(Lambda_P,1)'
    println("Lambda P Computed!\n")

    #Lambda B var(fe)
    Lambda_B_fe=sparse(rows,cols,Bii_fe,NT,NT)
    Lambda_B_fe=Lambda_B_fe+triu(Lambda_B_fe,1)'
    println("Lambda B FE Computed!\n")

    #Lambda B cov(fe,pe)
    if settings.cov_effects == true
        Lambda_B_cov=sparse(rows,cols,Bii_cov,NT,NT)
        Lambda_B_cov=Lambda_B_cov+triu(Lambda_B_cov,1)'
    end

    #Lambda B, var(pe)
    if settings.person_effects == true
        Lambda_B_pe=sparse(rows,cols,Bii_pe,NT,NT)
        Lambda_B_pe=Lambda_B_pe+triu(Lambda_B_pe,1)'
    end

    
    if settings.person_effects == false & settings.cov_effects == false
        return (Lambda_P = Lambda_P, Lambda_B_fe=Lambda_B_fe)
    elseif settings.person_effects == true & settings.cov_effects == false
        return (Lambda_P = Lambda_P, Lambda_B_fe=Lambda_B_fe, Lambda_B_pe=Lambda_B_pe)
    elseif settings.person_effects == true  & settings.cov_effects == true
        return (Lambda_P = Lambda_P, Lambda_B_fe=Lambda_B_fe, Lambda_B_pe=Lambda_B_pe, Lambda_B_cov=Lambda_B_cov)
    end

end


function eff_res(lev::JLAAlgorithm, X,id,firmid,match_id, K, settings)

    #Indexing Observations
    if settings.clustering_level == "obs"
        elist = index_constr(collect(1:length(id)), id, match_id )
    elseif settings.clustering_level == "match"
        elist = index_constr(match_id, id, match_id)
    else 
        error("Clustering level not possible to implement. \nOptions available are obs or match.")
    end
    
    #Dimensions
    NT=size(X,1)
    M=size(elist,1)
    J = maximum(firmid)
    N = maximum(id)
    p = lev.num_simulations == 0 ? ceil(log2(NT)/0.005) : lev.num_simulations
        
    #Initialize output
    Pii=zeros(M)
    Bii_fe=zeros(M)
    Bii_cov= settings.cov_effects ==true ? zeros(M) : nothing
    Bii_pe= settings.person_effects == true ? zeros(M) : nothing

    #No controls case
    if K == 0

        #Compute Auxiliaries
        movers , T = compute_movers(id, firmid)

        Nmatches = maximum(match_id)
        match_id_movers = [match_id[x] for x in findall(x->x==true, movers)]
        firmid_movers = [firmid[x] for x in findall(x->x==true, movers)]
        id_movers = [id[x] for x in findall(x->x==true, movers)]
        
        sel = unique(z -> match_id_movers[z], 1:length(match_id_movers))
        match_id_movers=match_id_movers[sel]
        firmid_movers=firmid_movers[sel]
        id_movers=id_movers[sel]
        
        maxT = maximum([T[x] for x in findall(x->x == false, movers)])
        
        counter = ones(Int,NT)
        gcs = Int.(@transform(groupby(DataFrame(counter = counter, id = id), :id), gcs = cumsum(:counter)).gcs)
        sel_stayers=(gcs.==1).*(movers.==false)
        stayers_matches_sel=[match_id[z] for z in findall(x->x == true , sel_stayers)]
        Tinv=1 ./T
        elist_JLL=[id_movers N.+firmid_movers id_movers N.+firmid_movers]

        M=size(elist_JLL,1)
        Pii_movers=zeros(M,1)
        Bii_fe_movers=zeros(M,1)
        Bii_cov_movers= settings.cov_effects ==true ? zeros(M,1) : nothing
        Bii_pe_movers= settings.person_effects == true ? zeros(M,1) : nothing

        #Initializing dependent variables for LSS
        AUX = initialize_auxiliary_variables(settings.leverage_algorithm, X, elist_JLL, M, NT, N, J, K, settings)

        for i=1:p

            OUT = compute_leverages(settings.leverage_algorithm, X, AUX, M, NT, N, J  , settings; scale= p)

            Pii_movers = OUT.Pii .+ Pii_movers
            Bii_fe_movers = OUT.Bii_fe .+ Bii_fe_movers
            Bii_pe_movers = Bii_pe == nothing ? nothing : OUT.Bii_pe .+ Bii_pe_movers
            Bii_cov_movers = Bii_cov_movers == nothing ? nothing : OUT.Bii_cov .+ Bii_cov_movers

        end

        #Assign Step
        Pii_movers=sparse(match_id_movers,ones(Int,length(match_id_movers)),Pii_movers[:,1],Nmatches,1)
        Pii_stayers=sparse(stayers_matches_sel,ones(Int,length(stayers_matches_sel)),[Tinv[x] for x in findall(x->x==true,sel_stayers)],Nmatches,1)
        Pii=Pii_movers.+Pii_stayers

        Bii_fe=sparse(match_id_movers,ones(Int,length(match_id_movers)),Bii_fe_movers[:,1],Nmatches,1)

        if settings.cov_effects == true
            Bii_cov=sparse(match_id_movers,ones(Int,length(match_id_movers)),Bii_cov_movers[:,1],Nmatches,1)
        end

        if settings.person_effects == true
            Bii_pe=sparse(match_id_movers,ones(Int,length(match_id_movers)),Bii_pe_movers[:,1],Nmatches,1)
            stayers = .!movers

            #Threads.@threads for t=2:maxT #T=1 have Pii=1 so need to be dropped.
            for t=2:maxT
                sel=(gcs.==true).*stayers.*(T.==t)
                N_sel=sum(sel)
                if N_sel > 0
                    index_sel=findall(x->x==true,sel)
                    match_sel_aux=Int.([match_id[z] for z in index_sel])
                    first=index_sel[1]
                    Xuse=X[first,:]

                    ztilde =lss(settings.lls_algorithm, X, Xuse, settings)
                    
                    aux_right=ztilde[1:N]
                    aux_left=ztilde[1:N]

                    COV=cov(X[:,1:N]*aux_left,X[:,1:N]*aux_right)
                    Bii_pe_stayers=COV[1]*(NT-1)

                    Bii_pe_stayers=sparse(match_sel_aux,ones(Int,length(match_sel_aux)),Bii_pe_stayers,Nmatches,1)
                    Bii_pe=Bii_pe.+Bii_pe_stayers
                end
            end    
        end
   

   

    elseif K> 0 
        #AUX = initialize_auxiliary_variables(settings.lls_algorithm, X, elist, M,NT, N, J, K, settings)
        nparameters = N + J + K

        D=sparse(collect(1:NT),id,1)
        F=sparse(collect(1:NT),firmid,1)
        Dvar = hcat(  D, spzeros(NT,nparameters-N) )
        Fvar = hcat(spzeros(NT,N), F, spzeros(NT,nparameters-N-J) )
        #Wvar = hcat(spzeros(NT,N+J), controls ) 
        Xleft = X[elist[:,1],:]
        Xright = X[elist[:,2],:]


    Threads.@threads for i=1:p
    #for i=1:2
    
        #Rademacher Entries
        rademach = rand(1,NT) .> 0.5
        rademach = rademach - (rademach .== 0)
        rademach = rademach ./sqrt(p)


            if settings.clustering_level == "obs"
                    Zleft = lss(settings.lls_algorithm, X, rademach*Xleft, settings)
                    #Zright = lss(settings.lls_algorithm, X, rademach*Xright, settings)

                    Pii = Pii .+ (X*Zleft).^2

                    rademach = rademach .- mean(rademach)

                    aux = lss(settings.lls_algorithm, X, rademach*Fvar, settings)
                    ZF = X*aux 
                    
                    Bii_fe = Bii_fe .+ ZF.^2 ./NT

                if settings.person_effects == true |    settings.cov_effects == true
                    aux = lss(settings.lls_algorithm, X, rademach*Dvar, settings)
                    ZD = X*aux
                end

                if settings.person_effects==true 
                    Bii_pe = Bii_pe .+ (ZD).^2 ./ NT
                end
                    
                if settings.cov_effects==true 
                    Bii_cov = Bii_cov .+ (ZD .* ZF ) ./ NT
                end
                    

            elseif settings.clustering_level == "match"

            end

        end


    end

    #Create matrices
    rows = elist[:,1]
    cols = elist[:,2]
    index_cluster = elist[:,3]

    if K==0
        Pii = [Pii[x] for x in index_cluster]
        Bii_fe = [Bii_fe[x] for x in index_cluster]

        println("Preparing to build Lambda matrices.\n")

        if settings.cov_effects == true
            Bii_cov = [Bii_cov[x] for x in index_cluster]
        end

        if settings.person_effects == true
            Bii_pe = [Bii_pe[x] for x in index_cluster]
        end

    end


    #Lambda P
    Lambda_P=sparse(rows,cols,Pii,NT,NT)
    Lambda_P=Lambda_P+triu(Lambda_P,1)'
    println("Lambda P Computed!\n")

    #Lambda B var(fe)
    Lambda_B_fe=sparse(rows,cols,Bii_fe,NT,NT)
    Lambda_B_fe=Lambda_B_fe+triu(Lambda_B_fe,1)'
    println("Lambda B FE Computed!\n")

    #Lambda B cov(fe,pe)
    if settings.cov_effects == true
        Lambda_B_cov=sparse(rows,cols,Bii_cov,NT,NT)
        Lambda_B_cov=Lambda_B_cov+triu(Lambda_B_cov,1)'
    end

    #Lambda B, var(pe)
    if settings.person_effects == true
        Lambda_B_pe=sparse(rows,cols,Bii_pe,NT,NT)
        Lambda_B_pe=Lambda_B_pe+triu(Lambda_B_pe,1)'
    end

    
    if settings.person_effects == false & settings.cov_effects == false
        return (Lambda_P = Lambda_P, Lambda_B_fe=Lambda_B_fe)
    elseif settings.person_effects == true & settings.cov_effects == false
        return (Lambda_P = Lambda_P, Lambda_B_fe=Lambda_B_fe, Lambda_B_pe=Lambda_B_pe)
    elseif settings.person_effects == true  & settings.cov_effects == true
        return (Lambda_P = Lambda_P, Lambda_B_fe=Lambda_B_fe, Lambda_B_pe=Lambda_B_pe, Lambda_B_cov=Lambda_B_cov)
    end

end

function check_clustering(clustering_var)
    NT = length(clustering_var)
    counter = ones(size(clustering_var,1));
    
    #Indexing obs number per worked/id
    gcs = Int.(@transform(groupby(DataFrame(counter = counter, clustering_var = clustering_var), :clustering_var), gcs = cumsum(:counter)).gcs)
    maxD = maximum(gcs)

    index = collect(1:NT)
    
    #This will be the output, and we will append observations to it in the loop
    list_final=DataFrame(row = Int64[],col = Int64[], id_cluster = Int64[])

    for t=1:maxD
        rowsel =  findall(x->x==true,gcs.==t)
        list_base = DataFrame( id_cluster= [clustering_var[x] for x in rowsel], row = 
        [index[x] for x in rowsel]  )
    
            for tt=t:maxD
                colsel =  findall(x->x==true,gcs.==tt)
                list_sel =  DataFrame( id_cluster= [clustering_var[x] for x in colsel], col = 
                [index[x] for x in colsel] )
    
                merge = outerjoin(list_base, list_sel, on = :id_cluster)
                merge = dropmissing(merge)
                merge = merge[:,[:row, :col, :id_cluster]]
    
                append!(list_final, merge)
    
            end
        end
    
        sort!(list_final, (:row));
    
        return (list_final = Matrix(list_final) , nnz_2 = size(list_final,1) )  

end

function do_Pii(X, clustering_var)
    n=size(X,1)

    #If no clustering, Lambda_P is just diagonal matrix.
    if clustering_var == nothing
        clustering_var = collect(1:n)
    end

    #Set matrices for parallel environment. 
    xx=X'*X
    P = aspreconditioner(ruge_stuben(xx))

    #Return the structure of the indexes associated with the clustering variable
    elist = check_clustering(clustering_var);
    M=size(elist,1)

    #Set elist 
    elist_1 = elist[:,1]
    elist_2 = elist[:,2]
    Pii=zeros(M,1)

    Threads.@threads for i=1:M
        zexact = zeros(size(X,2))
        col = elist_2[i]
        row = elist_1[i]
        cg!(zexact, xx, X[col,:], Pl = P , log=true, maxiter=300))

        Pii[i]=X[row,:]*zexact
    end        

    Lambda_P = sparse(elist[:,1],elist[:,2],Pii,n,n)

    return Lambda_P
end



function lincom_KSS(y,X,Z,Transform,clustering_var,Lambda_P; labels=nothing, restrict=nothing, nsim = 10000)
    #SET DIMENSIONS
    n=size(X,1)
    K=size(X,2)
    #Add Constant
    Z=hcat(ones(size(Z,1)), Z)  

    # PART 1: ESTIMATE HIGH DIMENSIONAL MODEL
    xx=X'*X
    xy=X'*y
    P = aspreconditioner(ruge_stuben(xx))
    beta = zeros(length(y))
    cg!(beta,xx , SparseMatrixCSC{Float64,Int64}(y), Pl = P  , log=true, maxiter=300)
    eta=y-X*beta
    
    # PART 1B: VERIFY LEAVE OUT COMPUTATION
    if Lambda_P == nothing 
        Lambda_P=do_Pii(X,clustering_var)
    end

    if Lambda_P != nothing && clusering_var !=nothing
        nnz_1=nnz(Lambda_P)
        nnz_2=check_clustering(clustering_var).nnz_2

        if nnz_1 == nnz_2
            println("The structure of the specified Lambda_P is consistent with the level of clustering required by the user.")
        elseif nnz_1 != nnz_2
            error("The user wants cluster robust inference but the Lambda_P provided by the user is not consistent with the level of clustering asked by the user. Try to omit input Lambda_P when running lincom_KSS")
        end
    end
    I_Lambda_P = I-Lambda_P
    eta_h = I_Lambda_P\eta

    #PART 2: SET UP MATRIX FOR SANDWICH FORMULA
    rows,columns, V = findnz(Lambda_P) 

    aux= 0.5*(y[rows].*eta_h[columns] + y[columns].*eta_h[rows]) 
    sigma_i=sparse(rows,columns,aux,n,n)

    aux= 0.5*(eta[rows].*eta[columns] + eta[columns].*eta[rows]) 
    sigma_i_res=sparse(rows,columns,aux,n,n)

    r=size(Z,2);
    wy=Transform*beta
    zz=Z'*Z

    numerator=Z\wy
    chet=wy-Z*numerator
    aux= 0.5*(chet[rows].*chet[columns] + chet[columns].*chet[rows]) 
    sigma_i_chet=sparse(rows,columns,aux,n,n)

    #PART 3: COMPUTE
    denominator=zeros(r,1)
    denominator_RES=zeros(r,1)

    for q=1:r
        v=sparse(q,1,1,r,1)
        v=zz\v
        v=Z*v
        v=Transform'*v

        right = zeros(length(v))
        cg!(right,xx , SparseMatrixCSC{Float64,Int64}(v), Pl = P  , log=true, maxiter=300)

        left=right' 

        denominator[q]=left*(X'*sigma_i*X)*right
        denominator_RES[q]=left*(X'*sigma_i_res*X)*right
    end   

    test_statistic=numerator./(sqrt(denominator))
    #zz_inv=zz^(-1)
    SE_linear_combination_NAI=zz\(Z'*sigma_i_chet*Z)/zz


    #PART 4: REPORT
    println("Inference on Linear Combinations:")
    if labels != nothing 
        for q=2:r
            if q <= r    
                println("Linear Combination - Column Number ", q-1," of Z: ", numerator[q] )
                println("Standard Error of the Linear Combination - Column Number ", q-1," of Z: ", sqrt(denominator[q]) )
                println("T-statistic - Column Number ", q-1, " of Z: ", test_statistic[q])
            end
        end
    else
        tell_me = labels[q-1]
        println("Linear Combination associated with ", tell_me,": ", numerator[q] )
        println("Standard Error  associated with ", tell_me,": ", sqrt(denominator[q]) )
        println("T-statistic  associated with ", tell_me,": ", test_statistic[q])
    end


    # PART 5: Joint-test. Quadratic form beta'*A*beta
    if restrict  == nothing 
        restrict=sparse(collect(1:r-1),collect(2:r),1,r-1,r)
    end

    v=restrict*(zz\(Z'*Transform))
    v=v'
    v=sparse(v)
    r=size(v,2)
    
    #Auxiliary
    aux=xx\v
    opt_weight=v'*aux
    opt_weight=opt_weight^(-1)
    opt_weight=(1/r)*(opt_weight+opt_weight')/2

    #Eigenvalues, eigenvectors, and relevant components
    FunAtimesX(x) = v*opt_weight*(v'*x)
    opts.issym = 1
    opts.tol = 1e-10
    [V, lambda]=eigs(FunAtimesX,K,xx,r,'largestabs',opts)
    lambda=diag(lambda)
    W=X*V
    V_b=W'*sigma_i*W


end



end