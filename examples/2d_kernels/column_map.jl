
# {{{
#        intma for DG indexing
#
function intma_dg(i, j, k, e)

    intma_dg = (e-1) * nglz * ngly * nglx + (k-1) * ngly * nglx + (j-1) * nglx + (i-1) + 1

    return intma_dg
    
end

function intma_1d_dg(k, e)

    return intma_1d_dg = (e - 1) * (nglz) + k
     
end


function node_column(nelz, nelem)
    
    do e = 1:nelem
        
        #calculate on-processor number of elements on a shell
        nelems = nelem / nelz
        
        #column and element numbering dependent code
        ecol = mod(e - 1 , nelems) + 1
        ie   = (e - 1) / nelems + 1
        endif
        
        for j = 1:ngly
            for i = 1:nglx
                
                ic = (ecol - 1) * ngly * nglx + (j - 1) * nglx + i
                
                for k = 1:nglz
                    iz = intma_1d_dg(k, ie)
                    
                    node_column_dg[iz, ic] = intma_dg(i, j, k, el)
                    
                end
            end
        end
    end
    return node_column
end
