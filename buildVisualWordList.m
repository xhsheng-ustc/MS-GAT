%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function buildVisualWordList()
%  visual code word and inverted list approach with kd-tree in fast dub detection
% revision history:
% z.li, started, 2010/04/05
% z.li, mbr bug, 2011/07/08
% input:
%   x - n x d data points
%   ht - kd-tree height
% output:
%   indx - indx structure with dim and val of cuts
%   leafs - leaf nodes of offs of x. 
%   mbrs - min bounding rectangles
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function [indx, leafs, mbrs]=buildVisualWordList(x, ht)
function [indx, leafs, mbrs]=buildVisualWordList(x, ht)
dbg =0;
if dbg
    x = randn(1024*4, 2); 
    ht = 4; 
end

% const
dbgplot=0;
if dbgplot
    styls = ['.r'; '.b'; '.k'; '.m'; '+r'; '+b'; '+k'; '+m'; '*r'; '*b'; '*k'; '*m']; 
    roffs = randperm(12); styls = styls(roffs, :); 
end

% var
[n, kd]=size(x); 
nNode       = 2^(ht+1) - 1; 
nLeafNode   = 2^ht;

% intermediate storages
offs = cell(nNode, 1); 

% first cut
indx.d_cuts(1) = 1; % cut at dimension 1
[sv, soffs]=sort(x(:,1)); 
indx.v_cuts(1) = sv(fix(n/2)); 
% split at mv, left child x(:,1) <= mv, right child x(:,1) > mv
offs{2} = soffs(1:fix(n/2)); 
offs{3} = soffs(fix(n/2)+1:n); 


for h=1:ht-1
    % parents nodes at height h, 
    for k=2^h:2^(h+1)-1
        % compute covariance
        offs_k = offs{k}; nk = length(offs_k); 
        % median offs 
        moffs = fix(nk/2); 
        sk = var(x(offs_k, :));
        [max_s, d_cut]=max(sk); 
        
        % cut dimension
        indx.d_cuts(k) = d_cut; 
        [sv, soffs]=sort(x(offs_k, d_cut));
        indx.v_cuts(k) = sv(moffs); 
        % current parent node k, left kid would be 2k, right kid would be
        % 2k+1
        offs{2*k}     = offs_k(soffs(1:moffs)); 
        offs{2*k+1}   = offs_k(soffs(moffs+1:nk)); 
        
        if (0)
            hold off; 
            plot(x(offs{k}, 1), x(offs{k}, 2), '.'); hold on;
            plot(x(offs{2*k}, 1), x(offs{2*k}, 2), '+r'); plot(x(offs{2*k+1}, 1), x(offs{2*k+1}, 2), '*k'); 
            
        end
        % prompt
        fprintf('\n split [%d: %d] at %d: %1.2f', k, nk, d_cut, sv(moffs)); 
                
        % clean up node k
        offs{k} = [];
    end
end

% leaf nodes
for j=1:nLeafNode
    leafs{j} = sort(offs{2^ht+j-1});
    mbrs{j}.min = min(x(leafs{j}));
    mbrs{j}.max = max(x(leafs{j}));
    if dbgplot
        grid on;
        if kd == 2
            plot(x(leafs{j},1), x(leafs{j}, 2), styls(mod(j, 12)+1,:)); hold on;
            plotbox(mbrs{j}.min, mbrs{j}.max, '-b');
        else
            plot3(x(leafs{j}, 1), x(leafs{j}, 2),x(leafs{j}, 3), styls(mod(j, 12)+1,:)); hold on;
        end
    end
end


return;
