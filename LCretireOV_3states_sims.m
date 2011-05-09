%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%	STOCHASTIC LIFE-CYCLE MODEL SOLVED WITH DYNAMIC PROGRAMMING 
%       Program exams how the decision to retire responds to 
%       stochastic asset price 
%
%    max \sum_{t=0}^T u(c_t,leisure)
%    
%    subject to
%
%    a_{t+1} = (1+r_t)*a_t + y - c_t   for t <= R
%    a_{t+1} = (1+r_t)*a_t - c_t   for R < t <= T
%
%    r_t follows a three-state Markov chain
%       - boom, bust, and crash
%
%  Rawley Heimer
%  Brandeis University
%  Spring 2011
%
%  w/ acknowledgement of George Hall for providing the foundation
%  of this code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all
diary lifecycle.out; 
disp('Rawley Heimer, CS177')
disp('This is my final project')
disp('STOCHASTIC LIFE-CYCLE MODEL SOLVED WITH DYNAMIC PROGRAMMING');
disp('This work analyzes how individuals make the decision to retire')
disp('when faced with the possibility of a crash in asset prices.')
disp('');

%-------------------------------------------------------------------------%
%                       set parameter values                              %
%-------------------------------------------------------------------------%

sigma   = 1.5;             % degree of relative risk aversion
beta    = 0.95;            % subjective discount factor 
theta   = 0.33;            % consumption/leisure elasticity
y       = 10;              % constant income stream  
r_low   = .08;                
r_high  = .12;             % rates of return 
r_crash = -.1;            
prob    = [.70 .25 .05; .25 .70 .05;.65 .25 .1]; % prob(i,j) = probability(s(t+1)=sj|s(t)=si)
lamda   = 0.3;             % SS payment as fraction of last paycheck.
leisure = .66;             % amount of leisure while working
T       = 25;              % number of periods in the agents life

%-------------------------------------------------------------------------%
%                       form wealth grid                                 %
%-------------------------------------------------------------------------%

maxw = 80;                         % maximum value of wealth grid   
minw = 0;                           % minimum value of wealth grid   
incw = .05;                         % size of wealth grid incremenats
wgrid = (minw:incw:maxw)';
nw = length(wgrid);                 % number of grid points

wp = repmat(wgrid,1,nw);
w  = repmat(wgrid,1,nw)';

%-------------------------------------------------------------------------% 
%  Calculate the utility function such that for zero or negative           %
%  consumption utility remains a large negative number so that            %
%  such values will never be chosen as utility maximizing                 %
%-------------------------------------------------------------------------%

cons_low = y + (1+r_low)*w - wp;
cons_high = y + (1+r_high)*w - wp;
cons_crash = y + (1+r_crash)*w - wp;
cons_retire_low = (1+r_low)*w + lamda*y- wp;
cons_retire_high = (1+r_high)*w + lamda*y -wp;
cons_retire_crash = (1+r_crash)*w + lamda*y -wp;

util_low    = (((cons_low.^theta)*(leisure.^(1-theta))).^(1-sigma) -1 )/(1-sigma);
util_high   = (((cons_high.^theta)*(leisure.^(1-theta))).^(1-sigma) -1 )/(1-sigma);
util_crash   = (((cons_crash.^theta)*(leisure.^(1-theta))).^(1-sigma) -1 )/(1-sigma);
util_retire_low = ((cons_retire_low.^theta).^(1-sigma) -1 )/(1-sigma);
util_retire_high = ((cons_retire_high.^theta).^(1-sigma) -1 )/(1-sigma);
util_retire_crash = ((cons_retire_crash.^theta).^(1-sigma) -1 )/(1-sigma);

util_low(cons_low<=0) = -inf;
util_high(cons_high<=0) = -inf;
util_crash(cons_crash<=0) = -inf;
util_retire_low(cons_retire_low<=0) = -inf;
util_retire_high(cons_retire_high<=0) = -inf;
util_retire_crash(cons_retire_crash<=0) = -inf;

clear cons_low cons_high cons_crash cons_retire_low cons_retire_high cons_retire_crash
%
%  initialize some variables
%
v = zeros(nw,3,T+1,T+1);        
tdecis   = zeros(nw,3,T,T);      
clear w wp
%
%  penalize the agent if he/she dies with negative wealth
%
v(wgrid< 0,:,T+1) = -inf;

%-------------------------------------------------------------------------%
% Use Backward Induction to Solve the Model: since a lifecycle model is 
% finite horizon model, just solve the Bellman equation backwards T
% periods.
%-------------------------------------------------------------------------%

format short g 

for RR=T:-1:1
for t=T:-1:RR+1

  [tv1,tdecis1]=max(util_retire_low + beta*repmat(v(:,:,t+1,RR)*prob(1,:)',1,nw));
  [tv2,tdecis2]=max(util_retire_high + beta*repmat(v(:,:,t+1,RR)*prob(2,:)',1,nw));
  [tv3,tdecis3]=max(util_retire_crash + beta*repmat(v(:,:,t+1,RR)*prob(3,:)',1,nw));
  
  tdecis(:,:,t,RR)=[tdecis1' tdecis2' tdecis3'];
  v(:,:,t,RR)=[tv1' tv2' tv3'];

end;

for t=RR:-1:1

  [tv1,tdecis1]=max(util_low + beta*repmat(v(:,:,t+1,RR)*prob(1,:)',1,nw));
  [tv2,tdecis2]=max(util_high + beta*repmat(v(:,:,t+1,RR)*prob(2,:)',1,nw));
  [tv3,tdecis3]=max(util_crash + beta*repmat(v(:,:,t+1,RR)*prob(3,:)',1,nw)); 
  
  tdecis(:,:,t,RR)=[tdecis1' tdecis2' tdecis3'];
  v(:,:,t,RR)=[tv1' tv2' tv3'];
  
end;
end

decis=(tdecis-1)*incw + minw;

clear tdecis1 tdecis2 tdecis3 tv1 tv2 tv3
%-------------------------------------------------------------------------%
% Use Backward Induction to Choose Retirement Date condtional on       %
% states at time t                                                        % 
%-------------------------------------------------------------------------%

t_work = 2;                         % number of periods forced to work
decRetire = zeros(nw,3,T+1);        % calculates decision rule
decRetire(:,:,1:t_work) = ones*2;   % force agent to work at start
decRetire(:,:,T) = ones;            % force agent to retire in T
                                    % in decRetire -- 2 = work; 1 = retire
for k = 1:length(prob(1,:))
    for i = t_work+1:T-1
        for j = 1:nw
        
        [value retire] = max([v(j,k,i,i),v(j,k,i,i+1)]);
        
        decRetire(j,k,i) = retire;
        end
    end
end

%-------------------------------------------------------------------------%
%    simulate life of the agent                                           %                                   
%-------------------------------------------------------------------------%

disp('SIMULATING LIFE HISTORY');
states   = zeros(T,2);
controls = zeros(T,2);
retired = 0;                        % the agent is not retired to start
s0 = 2;                             % initial state 
TT = 1000;                      % number of simultions

RetireYear = zeros(TT,1);       % Fill the retirement years.
states1    = zeros(T,TT);       % a_t
states2    = zeros(T,TT);       % r_t
controls1  = zeros(T,TT);       % c_t
controls2  = zeros(T,TT);       % a_t+1

for sim = 1:TT
wmark = find(wgrid==0);             
wealth = wgrid(wmark,1);            % initial level of assets
[chain,state] = markov(prob,T,s0);   
retired = 0; 
clear R

for i = 1:T;
    if chain(i) == 1;
        
        if retired == 0
            if decRetire(wmark,1,i) == 2
                R = i+1;
            elseif decRetire(wmark,1,i) == 1
                R = i;
            end
        elseif retired == 1
            R = R;
        end
        
       if R == i+1
           retired = 0;
       else
           retired = 1;
       end
        
       wealth_prime = decis(wmark,1,i,R);
       if i <= R;
     	  cons  = y + (1+r_low)*wealth - wealth_prime;
	      r = r_low;

       else
	      cons  = (1+r_low)*wealth + lamda*y - wealth_prime; 
	      r = r_low;

       end
       wmark = tdecis(wmark,1,i,R);

    elseif chain(i) == 2;
        
        if retired == 0
            if decRetire(wmark,2,i) == 2
                R = i+1;
            elseif decRetire(wmark,2,i) == 1
                R = i;
            end
        elseif retired == 1
            R = R;
        end
        
       if R == i+1
           retired = 0;
       else
           retired = 1;
       end
         
       wealth_prime = decis(wmark,2,i,R);
       if i <= R;
	      cons  = y + (1+r_high)*wealth - wealth_prime;
	      r = r_high;

       else
	      cons  = (1+r_high)*wealth + lamda*y - wealth_prime;
	      r = r_high;

       end
       wmark = tdecis(wmark,2,i,R);
      
    elseif chain(i) == 3;
        
        if retired == 0
            if decRetire(wmark,3,i) == 2
                R = i+1;
            elseif decRetire(wmark,3,i) == 1
                R = i;
            end
        elseif retired == 1
            R = R;
        end
        
       if R == i+1
           retired = 0;
       else
           retired = 1;
       end
         
       wealth_prime = decis(wmark,3,i,R);
       if i <= R;
	      cons  = y + (1+r_crash)*wealth - wealth_prime;
	      r = r_crash;

       else
	      cons  = (1+r_crash)*wealth + lamda*y - wealth_prime;
	      r = r_crash;

       end
       wmark = tdecis(wmark,3,i,R);
       
    else
      disp('something is wrong with chain');
    end;
    
    states(i,:) = [ wealth  r ];
    controls(i,:) = [ cons wealth_prime ];
    wealth = wealth_prime;
    
end;
    states1(:,sim) = states(:,1);
    states2(:,sim) = states(:,2);
    controls1(:,sim) = controls(:,1);
    controls2(:,sim) = controls(:,2);
    RetireYear(sim,1) = R;
end;

%-------------------------------------------------------------------------%
%   plot some info                                                        %
%-------------------------------------------------------------------------%

figure(1)
plot(R,0:.1:max(controls(:,1)),'color','b')
hold on
plot((1:T)',controls(:,1),'color','g')
hold on
plot((1:T)',states(:,2)*10,'color','r');
title('LIFE-CYCLE MODEL: RATE of RETURN, CONSUMPTION, & RETIREMENT DATE');
axis([ 1 (T) 0 1.5*y ])
legend('Retirement Year','Consumption','Rate of Return')
saveas(gcf,'Lifecycle','pdf')

figure(2)
plot(R,0:.1:max(controls(:,2)),(1:T)',controls(:,2));
title('LIFE CYCLE MODEL: SIMULATED WEALTH & RETIREMENT DATE');
saveas(gcf,'Wealth evolution','pdf')

figure(3)
hist(RetireYear)
title('Distribution of Retirement Years')
saveas(gcf,'RetireYrs','pdf')

%-------------------------------------------------------------------------%
%   display some statistics                                               %
%-------------------------------------------------------------------------%

disp('asset price returns')
disp('mean     std     skew')
disp([mean(prob*[r_low r_high r_crash]') std(prob*[r_low r_high r_crash]') skewness(prob*[r_low r_high r_crash]')]) 
disp(' ')
disp('distribution of retirement years')
disp('mean     std     skew')
disp([mean(RetireYear) std(RetireYear) skewness(RetireYear)])

diary off