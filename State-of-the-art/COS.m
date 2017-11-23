function distance = COS(data)

template_coupled_ca_stru = template_cate_coupling_computation(nominalProData); %similarity table
COD = template_coupled_ca_stru.COD;

function template_coupled_ca_stru = template_cate_coupling_computation(explicitView,hiddenView)
% ����template of coupled categorical attributes

Dim_explicit = size(explicitView,2);
View = [];
if nargin == 2
    View = [explicitView,hiddenView];
else
    View = explicitView;
end
Dim = size(View,2);
Unique_values_each_attribute = cell(1,Dim);
for jj = 1:Dim
    att = unique(View(:,jj));
    Unique_values_each_attribute{jj} = att;
end
template_coupled_ca_stru.Unique_values_each_attribute = Unique_values_each_attribute;
% Step I in tabletemplate - Intra-attribute coupling table based on Eq.(V.1)
template_coupled_ca_stru.Intra_coupling_table = intra_coupling(View,Unique_values_each_attribute);
% Step II tabletemplate - Intersection set (jj,kk)-element involves a 2D table based on the expression under Eq.(V.2)
% vk belongs to intersection set, phi_j_k(vjx) intersect phi_j_k(vjy)
template_coupled_ca_stru.Intersection_set = intersection_attributes(View,Unique_values_each_attribute);
% Step III & IV tabletemplate Eq.(V.2) - Inter_coupled_relative_similarity
% Dim*Dim �Խ����ǿռ�,each element corresponds to a value
template_coupled_ca_stru.Inter_coupled_relative_similarity_cell = ...
    inter_coupled_relative_similarity_cell(View,template_coupled_ca_stru.Intersection_set,Unique_values_each_attribute);

% Step V.1 ����coupling weights��ʱ�Ȳ�������
% 11-19 ������Բ���Can Wang's AAAI 2015���������£�����attribute������Ȩ���жϺ���
gammaArray = ones(1,(Dim-1)) / (Dim-1);% ��������һ������weights�ĺ���

% Step V tabletemplate - Eq.(V.3)
template_coupled_ca_stru.Inter_coupling_table = inter_coupling(Unique_values_each_attribute,Dim,...
    gammaArray,template_coupled_ca_stru.Inter_coupled_relative_similarity_cell );
% Step V.2 Eq.(V.4)
template_coupled_ca_stru.CCSC = coupled_ca_similarity(template_coupled_ca_stru.Intra_coupling_table,...
    template_coupled_ca_stru.Inter_coupling_table,Dim_explicit);
template_coupled_ca_stru.COD = coupled_ca_distance(template_coupled_ca_stru.Intra_coupling_table,...
    template_coupled_ca_stru.Inter_coupling_table,Dim_explicit);

% �ڹ�ʽ(V.5)�п������ޱ�Ҫ����ĥһ�½�1/L�滻Ϊһ��Ȩ��

% ***********

function CCSC = coupled_ca_similarity(Intra_coupling_table,Inter_coupling_table,Dim_explicit)
% Eq.(V.4)
CCSC = cell(1,Dim_explicit);
% Intra_coupling_table = cell(1,Dim);
% Inter_coupled_similarity_cell = cell(1,Dim);
for jj = 1:Dim_explicit   %����Ͱ�hidden view��Ӱ������ס�ˣ�
                          %�����ڼ���continuous to discrete coupling��ʱ��
                          %��continous-binning������Ϊhidden view���Լ��㲢�洢��
                          %���ǲ��ܼӵ���Ӧ��CCSC��ȥ��Ҳ����˵���ص�ֵ���ޣ�����hidden view��Ӱ��ӽ�ȥ��
    tableIntra = Intra_coupling_table{jj};
    tableInter = Inter_coupling_table{jj};
    tableIC = tableIntra.*tableInter;% ע�������ǵ�ˣ�����2D table�еĶ�ӦԪ�����
    CCSC{jj} = tableIC;
end


function COD = coupled_ca_distance(Intra_coupling_table,Inter_coupling_table,Dim_explicit)
% Eq.(V.4)
COD = cell(1,Dim_explicit);
% Intra_coupling_table = cell(1,Dim);
% Inter_coupled_similarity_cell = cell(1,Dim);
for jj = 1:Dim_explicit   %����Ͱ�hidden view��Ӱ������ס�ˣ�
                          %�����ڼ���continuous to discrete coupling��ʱ��
                          %��continous-binning������Ϊhidden view���Լ��㲢�洢��
                          %���ǲ��ܼӵ���Ӧ��CCSC��ȥ��Ҳ����˵���ص�ֵ���ޣ�����hidden view��Ӱ��ӽ�ȥ��
    tableIntra = Intra_coupling_table{jj};
    tableInter = Inter_coupling_table{jj};
    tableIC = (1./tableIntra-1).*(1-tableInter);% ע�������ǵ�ˣ�����2D table�еĶ�ӦԪ�����
    COD{jj} = tableIC;
end
% ���� matrix ��ʱ��
% ���������observationsʱ��,��template_coupled_ca_stru.CCSC����
% ͬʱִ������Ĵ���
function delta_IaO = IaOSO(View,CCSC,Dim_explicit)
N = size(View,1);
delta_IaO = zeros(N,N);
for xx = 1:(N-1)    
    for yy = (xx+1):N 
        delta_IaO_xx_yy = [];
        for jj = 1:Dim_explicit
            vjx = View(xx,jj);
            vjy = View(yy,jj);
            tableIC = CCSC{jj};
            delta_IaO_xx_yy = [delta_IaO_xx_yy;tableIC(vjx,vjy)];
        end
        % Ҳ��������������Ȩ�ؼ��㹫ʽ
        delta_IaO(xx,yy) = mean(delta_IaO_xx_yy);
    end
end



function Inter_coupled_similarity_cell = inter_coupling(Unique_values_each_attribute,Dim,gammaArray,Inter_coupled_relative_cell)
% Inter_coupled_relative_cell - Dim * Dim cell array
Inter_coupled_similarity_cell = cell(1,Dim); % i.e., delta_j_IeC
allindex = 1:Dim;
for jj = 1:Dim
    table_j_IeC = [];
    ind_left = setdiff(allindex,jj);
    attr = Unique_values_each_attribute{jj};
    for vjx = 1:length(attr)
        for vjy = 1:length(attr)  
            delta_j_IeC_vx_vy = [];
            for tt = 1:length(ind_left) %��kk���мӺͲ���
                kk = ind_left(tt);
                table = Inter_coupled_relative_cell{jj,kk};
                % Eq.(V.3)
                delta_j_IeC_vx_vy = [delta_j_IeC_vx_vy;gammaArray(tt) * table(vjx,vjy)];%�����������޸ģ�����Wang can's coupling weights
            end
            table_j_IeC(vjx,vjy) = sum(delta_j_IeC_vx_vy);%�洢����jj,kk�������refer to Eq.(V.3)��ע�������kk���мӺͲ���
        end
    end
    Inter_coupled_similarity_cell{jj} = table_j_IeC;
end




function Inter_coupled_relative_sim = inter_coupled_relative_similarity_cell(View,intersection_set,Unique_values)
% refer to Eq.(V.2)
% Inter_coupled_relative_sim = cell(Dim,Dim);
Dim = size(View,2);
allindex = [1:Dim];
Inter_coupled_relative_sim = cell(Dim,Dim);
for jj = 1:Dim
    attr = Unique_values{jj};
    ind_left = setdiff(allindex,jj);% ��������k~=j
    for tt = 1:length(ind_left)
        kk = ind_left(tt); 
        table = intersection_set{jj,kk} ;
        table_kk = zeros(length(attr));
        for vjx = 1:length(attr)
            for vjy = 1:length(attr)
                vkz_set = table{vjx,vjy};
                Min_x_y = [];
                for zz = 1:length(vkz_set)                    
                    vkz = vkz_set(zz);
                    % ICPx and ICPy ����information_conditional_probability table
                    % for each attribute
                    ICPx = information_conditional_probability(View,vkz,kk,vjx,jj);% Eq.(III.1) in conference paper
                    ICPy = information_conditional_probability(View,vkz,kk,vjy,jj);% Eq.(III.1) in conference paper
                    Min_x_y = [Min_x_y;min(ICPx,ICPy)];                    
                end
                table_kk(vjx,vjy) = sum(Min_x_y);% corresponding to Eq.(V.2)��%�ڸ���kk�к�jj�������£��洢���е�vjx��vjy�Ŀ�����
            end
        end
        Inter_coupled_relative_sim{jj,kk} = table_kk;%�洢���е�j|k������
    end
end


function ICPx = information_conditional_probability(S,vkz,kk,vjx,jj)
% �������������Ϊ����entry����
% compute information conditional probability Eq.(III.1)
ind_g_k_z = (S(:,kk)==vkz);% ii
ind_g_j_x = (S(:,jj)==vjx);% ii
fenzi = sum(ind_g_k_z & ind_g_j_x);
fenmu = sum(ind_g_j_x);
if fenmu ~= 0
    ICPx = fenzi/fenmu;
else
    disp('����information conditional probabilityʱ���ִ���');
    ICPx = -1;
end




function Intersection_set = intersection_attributes(View,Unique_values)
% Intersection set (jj,kk)-element involves a 2D table (Step II tabletemplate)
Dim = size(View,2);
Intersection_set = cell(Dim,Dim);
allindex = [1:Dim];
for jj = 1:Dim
    attr = Unique_values{jj};
    ind_left = setdiff(allindex,jj);
    for tt = 1:length(ind_left)
        kk = ind_left(tt);        
        % ����ÿ��element��Ӧ��2D Table
        table = cell(length(attr),length(attr));
        for vjx = 1:length(attr)
            for vjy = 1:length(attr)
                ind_g_j_x = (View(:,jj) == vjx);%�൱��u��indices, see page3������
                ind_g_j_y = (View(:,jj) == vjy);                
                phi_j_k_x = unique(View(ind_g_j_x,kk));%�ο�������phi4-2(alpha)={A,B} 
                phi_j_k_y = unique(View(ind_g_j_y,kk));
                table{vjx,vjy} = intersect(phi_j_k_x,phi_j_k_y);%�ο�������Eq.(V.2)����Ĺ�ʽphi_j_k(vjx) intersect phi_j_k(vjy)
            end
        end
        Intersection_set{jj,kk} = table;
    end
end

function Intra_coupling_table = intra_coupling(View,Unique_values)
% Intra-attribute coupling table (Step I in tabletemplate)
Dim = size(View,2);
Intra_coupling_table = cell(1,Dim);
for jj = 1:Dim
    attr = Unique_values{jj};
    Intra_table = zeros(length(attr));
    % ����һ��Ҫע��,����Ǿ���categorical��double������,֮�󾭹�ѭ��
    % �������
    % 1-L������,һ��Ҫ�ο�E:\mywork\coupled_multiview_clustering\pre_processing_cell2num.m
    % - lines 128-131
    % ������������������Ϊtemplate matrix�ľ���Index�������
    % ���磬Intra_table(vjx,vjy) = sim_value,��Ӧ��delta_jIac(vjx,vjy)����ֵ
    % �����ǰ�attr�е������Ű���˳�����°���λ��
    for vjx = 1:length(attr) 
        for vjy = 1:length(attr)
            Intra_table(vjx,vjy) = intra_coupled_similarity(View,vjx,vjy,jj);
        end
    end    
    Intra_coupling_table{jj} = Intra_table;
end




function delta_intra = intra_coupled_similarity(S,vjx,vjy,jj)
% delta_Intra - intra coupled similarity between cluster labels vjx and vjy
U1 = sum(S(:,jj) == vjx);
U2 = sum(S(:,jj) == vjy);
delta_intra = (U1 * U2)/(U1 + U2 + U1 * U2);





