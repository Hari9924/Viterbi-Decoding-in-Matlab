% Define constellation and parameters for both 4-PAM and 4-QAM
constellation_type = '4-QAM'; % Change to '4-PAM' or '4-QAM' based on requirement

switch constellation_type
    case '4-PAM'
        symbols = [-3, -1, 1, 3];
        bit_to_symbol_map = [-3, -1, 1, 3]; % Mapping for 4-PAM (00 -> -3, 01 -> -1, 10 -> +1, 11 -> +3)
        restricted_symbol = -3; % Symbol restriction for the final stages in 4-PAM
        num_bits_per_symbol = 2;
    case '4-QAM'
        symbols = [-1-1j, -1+1j, 1-1j, 1+1j];
        bit_to_symbol_map = [-1-1j, -1+1j, 1-1j, 1+1j]; % Mapping for 4-QAM
        restricted_symbol = -1-1j; % Symbol restriction for the final stages in 4-QAM
        num_bits_per_symbol = 2;
    otherwise
        error('Unsupported constellation type');
end

num_states = length(symbols)^2;

% Define the channel transfer function coefficients
h = [0.6/sqrt(2), -1/sqrt(2), 0.8/sqrt(2)];

% Generate all possible states (pairs of symbols) for 2 memory states
states = combvec(symbols, symbols)'; % (symbol1, symbol2) pairs

% Convert complex numbers to strings in readable format for state names
state_names = arrayfun(@(a, b) sprintf('(%s,%s)', complex_to_str(a), complex_to_str(b)), states(:, 1), states(:, 2), 'UniformOutput', false);

% Sort the states for consistency
[~, sort_idx] = sortrows(states, [1, 2]);
states = states(sort_idx, :);
state_names = state_names(sort_idx);

%% State Diagram
% Initialize the graph for the state diagram
G_state = digraph;
inputs_outputs = strings(0);

% Add all states as nodes in the graph
for i = 1:num_states
    G_state = addnode(G_state, state_names{i});
end

% Add transitions with input/output labels
for i = 1:num_states
    current_state = states(i, :);
    for input_symbol = symbols
        % Calculate the next state
        new_state = [input_symbol, current_state(1)];
        
        % Find the index of the new state
        next_state_idx = find(ismember(states, new_state, 'rows'));
        
        % Calculate the output using the channel transfer function
        output = input_symbol * h(1) + current_state(1) * h(2) + current_state(2) * h(3);
        
        % Format output as readable string
        output_str = complex_to_str(output);
        
        % Create the input/output label for this transition
        input_output_label = sprintf('%s/%s', complex_to_str(input_symbol), output_str);
        inputs_outputs(end + 1) = input_output_label;
        
        % Add edge with the custom label
        G_state = addedge(G_state, state_names{i}, state_names{next_state_idx});
    end
end

% Plot the state diagram with circular layout
figure;
p_state = plot(G_state, 'Layout', 'circle', 'NodeLabel', G_state.Nodes.Name);
title(sprintf('%s State Diagram with Circular Layout', constellation_type));
xlabel('States');
ylabel('Transitions');

% Set custom edge labels with input/output
p_state.EdgeLabel = inputs_outputs;
p_state.MarkerSize = 7;
p_state.NodeColor = 'cyan';
p_state.EdgeFontSize = 8;
p_state.NodeFontSize = 8;
p_state.LineWidth = 1;
p_state.EdgeColor = [0, 0, 0];  
p_state.ArrowSize = 10;

%% Trellis Diagram
% Define the number of stages (time steps)
num_stages = 5;
     
% Generate random input bits for the transmission
input_bits = randi([0, 1], 1, num_bits_per_symbol * num_stages);

% Convert bits to symbols using the bit-to-symbol map
input_symbols = bit_to_symbol_map(bi2de(reshape(input_bits, num_bits_per_symbol, []).', 'left-msb') + 1);

% Initialize the graph for the trellis diagram
G_trellis = digraph;
initial_state = 1; % Index of the initial state
initial_state_name = sprintf('Stage1_%s', state_names{initial_state});
G_trellis = addnode(G_trellis, initial_state_name);
inputs_outputs_trellis = strings(0);

% Define the SNR value (in dB)
SNR_dB = 0;
SNR_linear = 10^(SNR_dB / 10);
%SNR_linear = Eb/N0
noise_variance = 1 / SNR_linear;

% Incrementally build the trellis
for stage = 1:num_stages
    start_index = 8 + floor(log10(stage));
    current_stage_nodes = findnode(G_trellis, strcat('Stage', num2str(stage), '_', state_names));
    
    for i = current_stage_nodes'
        if i == 0, continue; end
        
        [~, state_idx] = ismember(G_trellis.Nodes.Name{i}(start_index:end), state_names);
        current_state = states(state_idx, :);
        a = current_state(1);
        b = current_state(2);
        
        % Apply restriction in the final stages
        if stage >= num_stages - 1
            current_symbols = restricted_symbol;
        else
            current_symbols = symbols;
        end
        
        for input_symbol = current_symbols
            new_state = [input_symbol, a];
            next_state_idx = find(ismember(states, new_state, 'rows'));
            if isempty(next_state_idx), continue; end
            next_state_name = sprintf('Stage%d_%s', stage + 1, state_names{next_state_idx});
            
            output = input_symbol * h(1) + a * h(2) + b * h(3);
            
            % Format output as readable string
            output_str = complex_to_str(output);
            
            input_output_label = sprintf('%s/%s', complex_to_str(input_symbol), output_str);
            inputs_outputs_trellis(end + 1) = input_output_label;
            
            if ~ismember(next_state_name, G_trellis.Nodes.Name)
                G_trellis = addnode(G_trellis, next_state_name);
            end
            
            source_node = G_trellis.Nodes.Name{i};
            target_node = next_state_name;
            G_trellis = addedge(G_trellis, source_node, target_node);
        end
    end
end

% Plot the trellis
figure;
p_trellis = plot(G_trellis, 'Layout', 'layered', 'NodeLabel', G_trellis.Nodes.Name);
title(sprintf('Trellis Diagram for %s', constellation_type));
xlabel('Stages');
ylabel('States');
p_trellis.EdgeLabel = inputs_outputs_trellis;

x_offsets = 0:num_stages;
y_values = linspace(-num_states/2, num_states/2, num_states);

for stage = 1:num_stages + 1
    stage_nodes = findnode(G_trellis, strcat('Stage', num2str(stage), '_', state_names));
    stage_nodes = stage_nodes(stage_nodes > 0);
    for idx = 1:length(stage_nodes)
        p_trellis.XData(stage_nodes(idx)) = x_offsets(stage);
        p_trellis.YData(stage_nodes(idx)) = y_values(idx);
    end
end

p_trellis.MarkerSize = 7;
p_trellis.NodeColor = 'cyan';
p_trellis.EdgeFontSize = 8;
p_trellis.NodeFontSize = 8;
p_trellis.LineWidth = 1;
p_trellis.EdgeColor = [0, 0, 0];
p_trellis.ArrowSize = 0.4;
%%
% Define the constellation type
constellation_type = '4-QAM'; % Change to '4-PAM' or '4-QAM' based on requirement

% Define symbols and memory states based on the constellation type
switch constellation_type
    case '4-PAM'
        symbols = [-3, -1, 1, 3];
    case '4-QAM'
        symbols = [-1-1j, -1+1j, 1-1j, 1+1j];
    otherwise
        error('Unsupported constellation type');
end

num_states = length(symbols)^2;

% Define the channel transfer function coefficients
h = [0.6/sqrt(2), -1/sqrt(2), 0.8/sqrt(2)];

% Generate all possible states (pairs of symbols) for 2 memory states
states = combvec(symbols, symbols)'; % (symbol1, symbol2) pairs
state_names = arrayfun(@(a, b) sprintf('(%s,%s)', num2str(a), num2str(b)), states(:, 1), states(:, 2), 'UniformOutput', false);

% Sort the states so that (-3, -3) is first and (3, 3) is last
[~, sort_idx] = sortrows(states, [1, 2]);
states = states(sort_idx, :);
state_names = state_names(sort_idx);

% Define the number of stages (time steps)
num_stages = 10000; % Large number to match Code 2

% Generate random input symbols for the transmission, with the last two set to a fixed symbol
input_symbols = symbols(randi(length(symbols), 1, num_stages));
input_symbols(num_stages) = symbols(1); % Restrict last two symbols for error-rate analysis
input_symbols(num_stages - 1) = symbols(1);

% Transmit signal through the channel
tx = input_symbols;
sk = conv(h, tx); % Convolution with channel
sk = sk(1:num_stages); % Truncate to match length

% Define the SNR values for the plot
SNR_dB_values = 2:0.25:16;
SER_values = zeros(size(SNR_dB_values));

% Loop over different SNR values
for idx = 1:length(SNR_dB_values)
    SNR_dB = SNR_dB_values(idx);
    SNR_linear = 10^(SNR_dB / 10);
    N0 = 1 / SNR_linear;
    %SNR_linear = Eb/N0
    noise_variance = sqrt(N0 / 2);
    
    % Generate and add noise to the transmitted signal
    noise = noise_variance * (randn(1, num_stages) + 1j * randn(1, num_stages));
    received_signal = sk + noise; % Received signal with noise
    
    % Viterbi Decoding
    cost = inf(num_stages + 1, num_states);   % Cost for each state at each stage
    prev_state = zeros(num_stages + 1, num_states); % Traceback for previous states
    cost(1, 1) = 0; % Start with zero cost at the initial state

    % Define the cost function for both real and imaginary parts
    calculateCost = @(rx, tx) abs(rx - tx)^2;

    % Perform Viterbi decoding with cost updates
    for t = 1:num_stages
        % Iterate over next states and find minimum-cost path
        for next_state = 1:num_states
            minTransitionCost = inf;
            bestPrevState = 0;

            % Consider all possible previous states
            for prev_state_idx = 1:num_states
                % Transition calculation
                current_state = states(prev_state_idx, :);
                next_input = states(next_state, 1); % Input symbol for next state
                predicted_output = h(1) * next_input + h(2) * current_state(1) + h(3) * current_state(2);

                % Calculate transition cost
                transition_cost = calculateCost(received_signal(t), predicted_output);
                cumulative_cost = cost(t, prev_state_idx) + transition_cost;

                % Update minimum cost and path if this path is cheaper
                if cumulative_cost < minTransitionCost
                    minTransitionCost = cumulative_cost;
                    bestPrevState = prev_state_idx;
                end
            end

            % Update cost and traceback if valid previous state found
            if bestPrevState > 0
                cost(t + 1, next_state) = minTransitionCost;
                prev_state(t + 1, next_state) = bestPrevState;
            end
        end
    end

    % Backtrack to find the most likely path and decoded symbols
    finalCosts = cost(num_stages + 1, :);
    [~, minEndIndex] = min(finalCosts);
    finalState = minEndIndex;

    decoded_symbols = zeros(1, num_stages);
    for t = num_stages:-1:1
        decoded_symbols(t) = states(finalState, 1);
        finalState = prev_state(t + 1, finalState);
    end

    % Calculate the Symbol Error Rate (SER)
    symbol_errors = sum(decoded_symbols ~= input_symbols);
    SER_values(idx) = (symbol_errors / num_stages);
    disp(SER_values);
end

% Plot SER vs. SNR
figure;
semilogy(SNR_dB_values, SER_values, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
xlabel('SNR (dB)');
ylabel('Symbol Error Rate (SER)');
title('SER vs. SNR 4-PAM');
ylim([10^(-2.5), 10^0]); % Set y-axis limits
grid on;


%% Helper function to format complex numbers as readable strings
function str = complex_to_str(x)
    if imag(x) >= 0
        str = sprintf('%.1f+%.1fj', real(x), imag(x));
    else
        str = sprintf('%.1f%.1fj', real(x), imag(x));
    end
end
