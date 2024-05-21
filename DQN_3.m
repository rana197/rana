% Start capturing the output to the file
diary();
indices = [];
range = 0:0.1:1;    %0.1 //0.05
n = length(range);
rewards_table = zeros(n, n, n, n); % Initialize rewards_table matrix // 4 Variable
%Instead reward table initialization, if you want to load from memory than
%loaded_data = load('E:\Research\sem_8\code\matlab\reward_table.mat');
%rewards = loaded_data.rewards;     % now, we got the reward table

% If you want to save rweward table than 
%save('E:\Research\sem_8\code\matlab\reward_table.mat', 'rewards');
% Assuming A_t (average latency) is calculated elsewhere and available
A_t = 20; % Placeholder value
% A_t_imaginary = 10000;

for W_d_index = 1:n
    for W_l_index = 1:n
        for W_ec_index = 1:n            
                for W_p_index = 1:n
                    W_d = range(W_d_index);
                    W_l = range(W_l_index);
                    W_ec = range(W_ec_index);    
                    W_p = range(W_p_index);                    
                    if W_d + W_l + W_ec  + W_p == 1
                        % Update rewards_table based on A_t
                        rewards_table(W_d_index, W_l_index, W_ec_index, W_p_index) = 1 / A_t;
                        indices = [indices; W_d_index, W_l_index, W_ec_index, W_p_index];
                        %else 
                        %rewards_table(W_d_index, W_l_index, W_ec_index) = 1 / A_t_imaginary;
                    end
            end
        end
    end
end

% Create a table from the indices array
index_rewards_table = array2table(indices, 'VariableNames', {'W_d_I', 'W_l_I', 'W_ec_I', 'W_p_I'});

% Initialize the replay buffer
bufferSize = 10000; % Maximum size of the buffer
%replayBuffer = struct('state', {}, 'action', {}, 'reward', {}, 'nextState', {}, 'done', {});
replayBuffer = struct('state', {}, 'action', {}, 'reward', {}, 'nextState', {});
RMSE = [];
loss_MSE = [];
inputSize = 4; % Assuming the state is represented by 3 values(W_distance, W_load, W_ec) now its 5
outputSize = 50; %Set of actions    // needs to be changed 10
net = createDeepQNetwork(inputSize, outputSize);        % Creating a Deep Q Network(called net)

episodes = 1; %100
gamma = 0.9; % Discount factor
epsilon = 0.5; % Exploration rate
alpha = 0.001; % Learning rate = or 0.01;
miniBatchSize = 5; % Size of the mini-batch

for episode = 1:episodes
    disp("current episode: " + episode);
    count = 1;   % will be a usefull flag for iteration/episode, initialize to 0/False at start of new episode
    % Get starting state
    [W_d_index, W_l_index, W_ec_index, W_p_index] = Get_Starting_State(index_rewards_table); 

    while (count <= 10)        %100
        disp("current count: " + count);
        state = [W_d_index, W_l_index, W_ec_index, W_p_index];

        predicted_action = Get_Predicted_Action(net,state, epsilon);

        % Take a step in the environment using the above predicted action.
        [new_W_d_Index, new_W_l_Index, new_W_ec_Index, new_W_p_Index] = Get_Next_State(W_d_index, W_l_index, W_ec_index, W_p_index, predicted_action);  
        
        % Calculate reward for jumping into new state using predicted action.
        new_index_reward = rewards_table(new_W_d_Index, new_W_l_Index, new_W_ec_Index,new_W_p_Index);
        
        % Calculate target Q-value: For each experience tuple, compute the target Q-value as reward + gamma * max(Q_values(next_state, :)),
        % where Q_values(next_state, :) are the Q-values predicted by the network for the next state.
        
        next_State = [new_W_d_Index new_W_l_Index new_W_ec_Index, new_W_p_Index];
        next_State_T = next_State';                                         % Transpose the state to make ready for DQN input.
        new_state_DL = dlarray(next_State_T, 'CB');                         % Converting the input(=nextState) into 
        next_Q_predict = predict(net, new_state_DL);                      % Exploit: Use the network to predict next Q-values
        % also need to assign "- infinity" to rejected actions for the next_State as well
        % add code here
        target_Q = new_index_reward + gamma * max(next_Q_predict, [], 1); % gamma is the discount factor, which represents the importance of future rewards relative to immediate rewards.
        
        % Add the experience to the replay buffer
        %addExperience(replayBuffer, state, predicted_action, target_Q, next_State, bufferSize,done);
        %addExperience(replayBuffer, state, predicted_action, target_Q, next_State, bufferSize); %done = 1     
        %adding experience using "addExperience" function , not working, so
        %adding experience in main loop (will debug later)
        if numel(replayBuffer) >= bufferSize
            replayBuffer(1) = []; % Remove the oldest experience if the buffer is full
        end
        %experience = struct('state', state, 'action', action, 'reward', reward, 'nextState', nextState, 'done', done);
        experience = struct('state', state, 'action', predicted_action, 'reward', target_Q, 'nextState', next_State);
        disp("latest experience:...");
        disp(experience);
        disp("adding past experience to replayBuffer");
        replayBuffer(end+1) = experience;
        disp("now, size of the replaybuffer is...");
        disp(size(replayBuffer));
        % Check if the buffer has enough samples to perform a training step
        %remainder = mod(x, y); % remainder will be 1 where x = 10, y = 1
        if ((numel(replayBuffer) >= miniBatchSize) && ~(mod(numel(replayBuffer), miniBatchSize)))
            % Sample a mini-batch of experiences from the replayBuffer
            [states, actions, rewards, nextStates] = sampleExperience(replayBuffer, miniBatchSize);           
            % Your existing code to perform a training step using the sampled mini-batch
            % Calculate target Q-values for the mini-batch
            targets = zeros(miniBatchSize, 1);
            for i = 1:miniBatchSize            
                targets(i) = rewards(i);               
                %next_State_DL = dlarray(nextStates(i, :)', 'CB');
                %next_Q_values = predict(net, next_State_DL);
                %targets(i) = rewards_table(nextStates(i), nextStates(i+2), nextStates(i+4)) + gamma * max(next_Q_values, [], 1);
            end        
        % Calculate loss and rmse
        [loss, gradients] = dlfeval(@modelGradients, net, states, targets, actions, miniBatchSize);
        loss_MSE = [loss_MSE; extractdata(loss)];  %saving loss batch wise 
        rmse = sqrt(loss);
        RMSE = [RMSE; extractdata(rmse)];          %saving rmse batch wise

        % Update the model/network
        disp("Updating the model/network...");
        net = dlupdate(@(w, dw) w + alpha * dw, net, gradients);
        end
     
        % Update state (setting: current state <- new state) OR % Move to the next state
        W_d_index = new_W_d_Index;
        W_l_index = new_W_l_Index;
        W_ec_index = new_W_ec_Index;
        W_p_index = new_W_p_Index;

        count = count +1;
    end
end
% Stop capturing the output
diary off;
function net = createDeepQNetwork(inputSize, outputSize)
    layers = [
        featureInputLayer(inputSize, 'Normalization', 'none', 'Name', 'input')
        fullyConnectedLayer(512, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(512, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(512, 'Name', 'fc3')
        reluLayer('Name', 'relu3')
        fullyConnectedLayer(512, 'Name', 'fc4')
        reluLayer('Name', 'relu4')
        fullyConnectedLayer(outputSize, 'Name', 'output')];
    lgraph = layerGraph(layers);
    net = dlnetwork(lgraph);
end

function [W_d_index, W_l_index, W_ec_index, W_p_index] = Get_Starting_State(index_rewards_table)

        % Randomly select indices for W_d, W_l, and W_ec
        numRows_I_table = size(index_rewards_table, 1);
        randomIndex = randi([1, numRows_I_table]);
        randomRow_I_table = index_rewards_table(randomIndex, :);
        W_d_I = randomRow_I_table(1, 1);
        W_d_index = W_d_I{1,1};
        W_l_I = randomRow_I_table(1, 2);
        W_l_index = W_l_I{1,1};
        W_ec_I = randomRow_I_table(1, 3);
        W_ec_index = W_ec_I{1,1};
        W_p_I = randomRow_I_table(1, 4);
        W_p_index = W_p_I{1,1};

end

%Action Selection: Define the function to select an action based on the current state and epsilon for exploration.

function predicted_action = Get_Predicted_Action(net,state, epsilon)
    validActions = [];

    for actionTry = 1:50
        if actionTry==1
            new_W_d_Index = state(1);  
            new_W_l_Index = state(2);  
            new_W_ec_Index = state(3)- 1; 
            new_W_p_Index = state(4) + 1;
        end
        if actionTry==2
            new_W_d_Index = state(1);
            new_W_l_Index = state(2);
            new_W_ec_Index = state(3)+ 1;
            new_W_p_Index = state(4) - 1;
        end
        if actionTry==3
            new_W_d_Index = state(1);
            new_W_l_Index = state(2) - 1;
            new_W_ec_Index = state(3);
            new_W_p_Index = state(4) + 1;
        end
        if actionTry==4
            new_W_d_Index = state(1);
            new_W_l_Index = state(2) + 1;
            new_W_ec_Index = state(3);
            new_W_p_Index = state(4) - 1;
        end
        if actionTry==5
            new_W_d_Index = state(1);
            new_W_l_Index = state(2) + 1;
            new_W_ec_Index = state(3) - 1;
            new_W_p_Index = state(4);
        end
        if actionTry==6
            new_W_d_Index = state(1);
            new_W_l_Index = state(2) - 1;
            new_W_ec_Index = state(3) + 1;
            new_W_p_Index = state(4);
        end        
        if actionTry==7 % decrement W_d & increment W_ec
            new_W_d_Index = state(1);
            new_W_l_Index = state(2) - 1;
            new_W_ec_Index = state(3)- 1;
            new_W_p_Index = state(4) + 2;
        end
        if actionTry==8 % decrement W_d & decrement W_ec
            new_W_d_Index = state(1);
            new_W_l_Index = state(2) + 1;
            new_W_ec_Index = state(3)+ 1;
            new_W_p_Index = state(4) - 2;
        end
        if actionTry==9 % Stay in W_d & increment W_l
            new_W_d_Index = state(1) - 1;
            new_W_l_Index = state(2) ;
            new_W_ec_Index = state(3);
            new_W_p_Index = state(4) + 1;
        end
        if actionTry==10 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1) + 1;
            new_W_l_Index = state(2) ;
            new_W_ec_Index = state(3);
            new_W_p_Index = state(4) - 1;
        end
        if actionTry==11 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1) -1;
            new_W_l_Index = state(2) ;
            new_W_ec_Index = state(3) + 1;
            new_W_p_Index = state(4);
        end
        if actionTry==12 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1) + 1;
            new_W_l_Index = state(2) ;
            new_W_ec_Index = state(3) - 1;
            new_W_p_Index = state(4);
        end
        if actionTry==13 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1) + 1;
            new_W_l_Index = state(2) ;
            new_W_ec_Index = state(3) + 1;
            new_W_p_Index = state(4) - 2;
        end
        if actionTry==14 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1) - 1;
            new_W_l_Index = state(2);
            new_W_ec_Index = state(3) - 1;
            new_W_p_Index = state(4) + 2;
        end
        if actionTry==15 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1)-1;
            new_W_l_Index = state(2)+1;
            new_W_ec_Index = state(3);
            new_W_p_Index = state(4);
        end
        if actionTry==16 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1)+1;
            new_W_l_Index = state(2)-1;
            new_W_ec_Index = state(3);
            new_W_p_Index = state(4);
        end
        if actionTry==17 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1)+1;
            new_W_l_Index = state(2)-2;
            new_W_ec_Index = state(3);
            new_W_p_Index = state(4)+1;
        end
        if actionTry==18 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1)+2;
            new_W_l_Index = state(2)-1;
            new_W_ec_Index = state(3);
            new_W_p_Index = state(4)-1;
        end
        if actionTry==19 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1)-2;
            new_W_l_Index = state(2)+1;
            new_W_ec_Index = state(3);
            new_W_p_Index = state(4)+1;
        end
        if actionTry==20 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1)-1;
            new_W_l_Index = state(2)+2;
            new_W_ec_Index = state(3);
            new_W_p_Index = state(4)-1;
        end
        if actionTry==21 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1) + 1;
            new_W_l_Index = state(2) + 1;
            new_W_ec_Index = state(3);
            new_W_p_Index = state(4) - 2;
        end
        if actionTry==22 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1) -1;
            new_W_l_Index = state(2) -1;
            new_W_ec_Index = state(3);
            new_W_p_Index = state(4) +2;
        end
        if actionTry==23 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1) +1;
            new_W_l_Index = state(2) +1;
            new_W_ec_Index = state(3) -2;
            new_W_p_Index = state(4);
        end
        if actionTry==24 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1) +1;
            new_W_l_Index = state(2) -2;
            new_W_ec_Index = state(3) +1;
            new_W_p_Index = state(4);
        end
        if actionTry==25 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1) -2;
            new_W_l_Index = state(2) +1;
            new_W_ec_Index = state(3) +1;
            new_W_p_Index = state(4);
        end
        if actionTry==26 
            new_W_d_Index = state(1) -1;
            new_W_l_Index = state(2) -1;
            new_W_ec_Index = state(3) +2;
            new_W_p_Index = state(4);
        end
        if actionTry==27 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1) +2;
            new_W_l_Index = state(2) -1;
            new_W_ec_Index = state(3)-1;
            new_W_p_Index = state(4);
        end
        if actionTry==28 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1) -1;
            new_W_l_Index = state(2) +2;
            new_W_ec_Index = state(3)-1;
            new_W_p_Index = state(4);
        end
        if actionTry==29 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1)+1;
            new_W_l_Index = state(2)+1;
            new_W_ec_Index = state(3)+1;
            new_W_p_Index = state(4)-3;
        end
        if actionTry==30 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1)+1;
            new_W_l_Index = state(2)+1;
            new_W_ec_Index = state(3)-1;
            new_W_p_Index = state(4)-1;
        end
        if actionTry==31 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1)+1;
            new_W_l_Index = state(2)-1;
            new_W_ec_Index = state(3)+1;
            new_W_p_Index = state(4)-1;
        end
        if actionTry==32 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1)+1;
            new_W_l_Index = state(2)-1;
            new_W_ec_Index = state(3)-1;
            new_W_p_Index = state(4)+1;
        end
        if actionTry==33 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1)-1;
            new_W_l_Index = state(2)+1;
            new_W_ec_Index = state(3)+1;
            new_W_p_Index = state(4)-1;
        end
        if actionTry==34 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1)-1;
            new_W_l_Index = state(2)+1;
            new_W_ec_Index = state(3)-1;
            new_W_p_Index = state(4)+1;
        end
        if actionTry==35 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1)-1;
            new_W_l_Index = state(2)-1;
            new_W_ec_Index = state(3)+1;
            new_W_p_Index = state(4)+1;
        end
        if actionTry==36 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1)-1;
            new_W_l_Index = state(2)-1;
            new_W_ec_Index = state(3)-1;
            new_W_p_Index = state(4)+3;
        end
        if actionTry==37 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1)+1;
            new_W_l_Index = state(2)+1;
            new_W_ec_Index = state(3)-3;
            new_W_p_Index = state(4)+1;
        end
        if actionTry==38 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1)+1;
            new_W_l_Index = state(2)-3;
            new_W_ec_Index = state(3)+1;
            new_W_p_Index = state(4)+1;
        end
        if actionTry==39 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1)-3;
            new_W_l_Index = state(2)+1;
            new_W_ec_Index = state(3)+1;
            new_W_p_Index = state(4)+1;
        end
        if actionTry==40 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1)-1;
            new_W_l_Index = state(2)-1;
            new_W_ec_Index = state(3)+3;
            new_W_p_Index = state(4)-1;
        end
        if actionTry==41 
            new_W_d_Index = state(1)-1;
            new_W_l_Index = state(2)+3;
            new_W_ec_Index = state(3)-1;
            new_W_p_Index = state(4)-1;
        end
        if actionTry==42 
            new_W_d_Index = state(1)+3;
            new_W_l_Index = state(2)-1;
            new_W_ec_Index = state(3)-1;
            new_W_p_Index = state(4)-1;
        end
        if actionTry==43 
            new_W_d_Index = state(1);
            new_W_l_Index = state(2)+1;
            new_W_ec_Index = state(3)-2;
            new_W_p_Index = state(4)+1;
        end
        if actionTry==44 
            new_W_d_Index = state(1);
            new_W_l_Index = state(2)-2;
            new_W_ec_Index = state(3)+1;
            new_W_p_Index = state(4)+1;
        end
        if actionTry==45
            new_W_d_Index = state(1);
            new_W_l_Index = state(2)-1;
            new_W_ec_Index = state(3)+2;
            new_W_p_Index = state(4)-1;
        end
        if actionTry==46 
            new_W_d_Index = state(1);
            new_W_l_Index = state(2)+2;
            new_W_ec_Index = state(3)-1;
            new_W_p_Index = state(4)-1;
        end
        if actionTry==47 
            new_W_d_Index = state(1)+1;
            new_W_l_Index = state(2);
            new_W_ec_Index = state(3)-2;
            new_W_p_Index = state(4)+1;
        end
        if actionTry==48 
            new_W_d_Index = state(1)-2;
            new_W_l_Index = state(2);
            new_W_ec_Index = state(3)+1;
            new_W_p_Index = state(4)+1;
        end
        if actionTry==49 
            new_W_d_Index = state(1)-1;
            new_W_l_Index = state(2);
            new_W_ec_Index = state(3)+2;
            new_W_p_Index = state(4)-1;
        end
        if actionTry==50 
            new_W_d_Index = state(1)+2;
            new_W_l_Index = state(2);
            new_W_ec_Index = state(3)-1;
            new_W_p_Index = state(4)-1;
        end
        % After calculating new_W_d_Index, new_W_l_Index, new_W_ec_Index
        % Check if any index is less than 1 or greater than 11
        if new_W_d_Index < 1 || new_W_l_Index < 1 || new_W_ec_Index < 1 || new_W_p_Index < 1 || new_W_d_Index > 11 || new_W_l_Index > 11 || new_W_ec_Index > 11 || new_W_p_Index > 11
            %disp("dont update validActions array...");
        else
            %disp("updating validActions array...");
            validActions = [validActions, actionTry];     % True = 1
        end
    end
    disp("current state: ");
    disp(state);
    disp("its Valid Actions: ");
    disp(validActions)

        % Exploration: choose a random valid action
        if rand() < epsilon
            disp("Exploring as, in this count, rand() < epsilon ");
            random_action_Index = randi(length(validActions));
            predicted_action = validActions(random_action_Index);
            predicted_action= dlarray(predicted_action);
        else
            disp("Exploiting as, in this count, rand() > epsilon ");
            rejected_Actions = setdiff(1:50, validActions);         % computing rejected_Actions, Assuming 10 possible actions
            disp("its rejected_Actions: ");
            disp(rejected_Actions)
            state = state';                                % Transpose the state to make ready for DQN input.
            state_DL = dlarray(state, 'CB');               % Converting the input(=state) into 
            Q_predict = predict(net, state_DL);            % Exploit: Use the network to predict Q-values
            Q_predict(rejected_Actions)=-inf;              % Initializing rejected_Actions as -Infinity.            
            
            % Check if all elements in Q_predict are -Inf
            all_elements_are_minus_inf = all(Q_predict == -Inf);
            %disp(all_elements_are_minus_inf); % This will be true
            if all_elements_are_minus_inf
                Q_predict(validActions)=-1000;               % Initializing validActions as -10 to avoid conflict if incase all values of Q_predict possess -Infinity(-inf)
            end
            
            disp("Printing predicted Q values:Q_predict...");
            disp(Q_predict);
            % Assign value -100 to cells that have NaN
            Q_predict(isnan(Q_predict)) = -1000;
            [pred_current_max_reward, predicted_action] = max(Q_predict, [], 1);   % Choose the action with the highest Q-value(=pred_current_max_reward)
            %disp("current predicted maximum reward: " + pred_current_max_reward);
        end
end

function [new_W_d_Index, new_W_l_Index, new_W_ec_Index,new_W_p_Index] = Get_Next_State(W_d_index, W_l_index, W_ec_index,W_p_index, action)
    % step = 0.1;
    % Action to state transition logic
    disp("choosen action from the above valid actions: ")
    disp(action)
    switch action     
        case 1 
            new_W_d_Index = W_d_index;  
            new_W_l_Index = W_l_index;  
            new_W_ec_Index = W_ec_index - 1; 
            new_W_p_Index = W_p_index + 1;
        case 2 
            new_W_d_Index = W_d_index;
            new_W_l_Index = W_l_index;
            new_W_ec_Index = W_ec_index + 1;
            new_W_p_Index = W_p_index - 1;
        
        case 3 
            new_W_d_Index = W_d_index;
            new_W_l_Index = W_l_index - 1;
            new_W_ec_Index = W_ec_index;
            new_W_p_Index = W_p_index + 1;
        
        case 4
            new_W_d_Index = W_d_index;
            new_W_l_Index = W_l_index + 1;
            new_W_ec_Index = W_ec_index;
            new_W_p_Index = W_p_index - 1;
        
        case 5
            new_W_d_Index = W_d_index;
            new_W_l_Index = W_l_index + 1;
            new_W_ec_Index = W_ec_index - 1;
            new_W_p_Index = W_p_index;

        case 6       
            new_W_d_Index = W_d_index;
            new_W_l_Index = W_l_index - 1;
            new_W_ec_Index = W_ec_index + 1;
            new_W_p_Index = W_p_index;

        case 7               
            new_W_d_Index = W_d_index;
            new_W_l_Index = W_l_index - 1;
            new_W_ec_Index = W_ec_index- 1;
            new_W_p_Index = W_p_index + 2;
        
        case 8
            new_W_d_Index = W_d_index;
            new_W_l_Index = W_l_index + 1;
            new_W_ec_Index = W_ec_index+ 1;
            new_W_p_Index = W_p_index - 2;
        
        case 9
            new_W_d_Index = W_d_index - 1;
            new_W_l_Index = W_l_index ;
            new_W_ec_Index = W_ec_index;
            new_W_p_Index = W_p_index + 1;
        
        case 10 
            new_W_d_Index = W_d_index + 1;
            new_W_l_Index = W_l_index ;
            new_W_ec_Index = W_ec_index;
            new_W_p_Index = W_p_index - 1;
        
        case 11
            new_W_d_Index = W_d_index -1;
            new_W_l_Index = W_l_index ;
            new_W_ec_Index = W_ec_index + 1;
            new_W_p_Index = W_p_index;
        
        case 12
            new_W_d_Index = W_d_index + 1;
            new_W_l_Index = W_l_index ;
            new_W_ec_Index = W_ec_index - 1;
            new_W_p_Index = W_p_index;
        
        case 13
            new_W_d_Index = W_d_index + 1;
            new_W_l_Index = W_l_index ;
            new_W_ec_Index = W_ec_index + 1;
            new_W_p_Index = W_p_index - 2;
        
        case 14
            new_W_d_Index = W_d_index - 1;
            new_W_l_Index = W_l_index;
            new_W_ec_Index = W_ec_index - 1;
            new_W_p_Index = W_p_index + 2;
        case 15 %.
            new_W_d_Index = W_d_index-1;
            new_W_l_Index = W_l_index+1;
            new_W_ec_Index = W_ec_index;
            new_W_p_Index = W_p_index;
        
        case 16 %.
            new_W_d_Index = W_d_index+1;
            new_W_l_Index = W_l_index-1;
            new_W_ec_Index = W_ec_index;
            new_W_p_Index = W_p_index;
        
        case 17 %.
            new_W_d_Index = W_d_index+1;
            new_W_l_Index = W_l_index-2;
            new_W_ec_Index = W_ec_index;
            new_W_p_Index = W_p_index+1;
        
        case 18 %.
            new_W_d_Index = W_d_index+2;
            new_W_l_Index = W_l_index-1;
            new_W_ec_Index = W_ec_index;
            new_W_p_Index = W_p_index-1;
        
        case 19 %.
            new_W_d_Index = W_d_index-2;
            new_W_l_Index = W_l_index+1;
            new_W_ec_Index = W_ec_index;
            new_W_p_Index = W_p_index+1;
        
        case 20 %.
            new_W_d_Index = W_d_index-1;
            new_W_l_Index = W_l_index+2;
            new_W_ec_Index = W_ec_index;
            new_W_p_Index = W_p_index-1;
        
        case 21 %.
            new_W_d_Index = W_d_index + 1;
            new_W_l_Index = W_l_index + 1;
            new_W_ec_Index = W_ec_index;
            new_W_p_Index = W_p_index - 2;
        
        case 22 %.
            new_W_d_Index = W_d_index -1;
            new_W_l_Index = W_l_index -1;
            new_W_ec_Index = W_ec_index;
            new_W_p_Index = W_p_index +2;
        
        case 23 %.
            new_W_d_Index = W_d_index +1;
            new_W_l_Index = W_l_index +1;
            new_W_ec_Index = W_ec_index -2;
            new_W_p_Index = W_p_index;
        
        case 24 %.
            new_W_d_Index = W_d_index +1;
            new_W_l_Index = W_l_index -2;
            new_W_ec_Index = W_ec_index +1;
            new_W_p_Index = W_p_index;
        
        case 25 %.
            new_W_d_Index = W_d_index -2;
            new_W_l_Index = W_l_index +1;
            new_W_ec_Index = W_ec_index +1;
            new_W_p_Index = W_p_index;
        
        case 26 
            new_W_d_Index = W_d_index -1;
            new_W_l_Index = W_l_index -1;
            new_W_ec_Index = W_ec_index +2;
            new_W_p_Index = W_p_index;
        
        case 27 %.
            new_W_d_Index = W_d_index +2;
            new_W_l_Index = W_l_index -1;
            new_W_ec_Index = W_ec_index-1;
            new_W_p_Index = W_p_index;
        
        case 28 %.
            new_W_d_Index = W_d_index -1;
            new_W_l_Index = W_l_index +2;
            new_W_ec_Index = W_ec_index-1;
            new_W_p_Index = W_p_index;
        
        case 29 %.
            new_W_d_Index = W_d_index+1;
            new_W_l_Index = W_l_index+1;
            new_W_ec_Index = W_ec_index+1;
            new_W_p_Index = W_p_index-3;
        
        case 30 %.
            new_W_d_Index = W_d_index+1;
            new_W_l_Index = W_l_index+1;
            new_W_ec_Index = W_ec_index-1;
            new_W_p_Index = W_p_index-1;
        
        case 31 %.
            new_W_d_Index = W_d_index+1;
            new_W_l_Index = W_l_index-1;
            new_W_ec_Index = W_ec_index+1;
            new_W_p_Index = W_p_index-1;
        
        case 32 %.
            new_W_d_Index = W_d_index+1;
            new_W_l_Index = W_l_index-1;
            new_W_ec_Index = W_ec_index-1;
            new_W_p_Index = W_p_index+1;
        
        case 33 %.
            new_W_d_Index = W_d_index-1;
            new_W_l_Index = W_l_index+1;
            new_W_ec_Index = W_ec_index+1;
            new_W_p_Index = W_p_index-1;
        
        case 34 %.
            new_W_d_Index = W_d_index-1;
            new_W_l_Index = W_l_index+1;
            new_W_ec_Index = W_ec_index-1;
            new_W_p_Index = W_p_index+1;
        
        case 35 %.
            new_W_d_Index = W_d_index-1;
            new_W_l_Index = W_l_index-1;
            new_W_ec_Index = W_ec_index+1;
            new_W_p_Index = W_p_index+1;
        
        case 36 %.
            new_W_d_Index = W_d_index-1;
            new_W_l_Index = W_l_index-1;
            new_W_ec_Index = W_ec_index-1;
            new_W_p_Index = W_p_index+3;
        
        case 37 %.
            new_W_d_Index = W_d_index+1;
            new_W_l_Index = W_l_index+1;
            new_W_ec_Index = W_ec_index-3;
            new_W_p_Index = W_p_index+1;
        
        case 38 %.
            new_W_d_Index = W_d_index+1;
            new_W_l_Index = W_l_index-3;
            new_W_ec_Index = W_ec_index+1;
            new_W_p_Index = W_p_index+1;
        
        case 39 %.
            new_W_d_Index = W_d_index-3;
            new_W_l_Index = W_l_index+1;
            new_W_ec_Index = W_ec_index+1;
            new_W_p_Index = W_p_index+1;
        
        case 40 %.
            new_W_d_Index = W_d_index-1;
            new_W_l_Index = W_l_index-1;
            new_W_ec_Index = W_ec_index+3;
            new_W_p_Index = W_p_index-1;
        
        case 41 
            new_W_d_Index = W_d_index-1;
            new_W_l_Index = W_l_index+3;
            new_W_ec_Index = W_ec_index-1;
            new_W_p_Index = W_p_index-1;
        
        case 42 
            new_W_d_Index = W_d_index+3;
            new_W_l_Index = W_l_index-1;
            new_W_ec_Index = W_ec_index-1;
            new_W_p_Index = W_p_index-1;
        
        case 43 
            new_W_d_Index = W_d_index;
            new_W_l_Index = W_l_index+1;
            new_W_ec_Index = W_ec_index-2;
            new_W_p_Index = W_p_index+1;
        
        case 44 %.
            new_W_d_Index = W_d_index;
            new_W_l_Index = W_l_index-2;
            new_W_ec_Index = W_ec_index+1;
            new_W_p_Index = W_p_index+1;
        
        case 45 %.
            new_W_d_Index = W_d_index;
            new_W_l_Index = W_l_index-1;
            new_W_ec_Index = W_ec_index+2;
            new_W_p_Index = W_p_index-1;
        
        case 46 
            new_W_d_Index = W_d_index;
            new_W_l_Index = W_l_index+2;
            new_W_ec_Index = W_ec_index-1;
            new_W_p_Index = W_p_index-1;
        
        case 47 
            new_W_d_Index = W_d_index+1;
            new_W_l_Index = W_l_index;
            new_W_ec_Index = W_ec_index-2;
            new_W_p_Index = W_p_index+1;
        
        case 48 
            new_W_d_Index = W_d_index-2;
            new_W_l_Index = W_l_index;
            new_W_ec_Index = W_ec_index+1;
            new_W_p_Index = W_p_index+1;
        
        case 49 
            new_W_d_Index = W_d_index-1;
            new_W_l_Index = W_l_index;
            new_W_ec_Index = W_ec_index+2;
            new_W_p_Index = W_p_index-1;
        
        case 50 
            new_W_d_Index = W_d_index+2;
            new_W_l_Index = W_l_index;
            new_W_ec_Index = W_ec_index-1;
            new_W_p_Index = W_p_index-1;
    end
end

% Function to sample a mini-batch of experiences from the buffer
%function [states, actions, rewards, nextStates, dones] = sampleExperience(replayBuffer, batchSize)
function [states, actions, rewards, nextStates] = sampleExperience(replayBuffer, batchSize)
    idx = randperm(numel(replayBuffer), batchSize);
    miniBatch = replayBuffer(idx);
    
    states = vertcat(miniBatch.state);
    actions = vertcat(miniBatch.action);
    rewards = vertcat(miniBatch.reward);
    nextStates = vertcat(miniBatch.nextState);
    %dones = vertcat(miniBatch.done);
end    

function [loss, gradients] = modelGradients(net, states, targets, actions, miniBatchSize)
    states_DL = dlarray(states', 'CB');
    predicted_Q_values = forward(net, states_DL);
    Q_values_for_actions = predicted_Q_values(sub2ind(size(predicted_Q_values), (extractdata(actions))', 1:miniBatchSize));
    %Q_values_for_actions = predicted_Q_values(sub2ind(size(predicted_Q_values), actions', 1:miniBatchSize));
    targets_dl =dlarray(targets', 'CT');
    loss = mse(targets_dl, Q_values_for_actions);       %datatype of "loss" should be dlarray. error otherwise.
    %loss = mse(extractdata(targets), extractdata(Q_values_for_actions));
    %loss = mse(targets, Q_values_for_actions);
    %loss = [0.3087 0.3087];
    gradients = dlgradient(loss, net.Learnables);
end

