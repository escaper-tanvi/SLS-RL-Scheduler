classdef customSchedulerRL < nrScheduler & rl.env.MATLABEnvironment
    properties (Access = public)
        UEOrder

        Agent          % RL agent

        oInfo % Observation space

        aInfo      % Action space

        UEThroughput

        numUEs

        EventData

        UsersPerTTI
    end

    events (Hidden)
        NewTransmissionRLScheduler
    end

    methods
        function obj = customSchedulerRL(numUEs, usersPerTTI)
            % Call MATLAB RL Environment constructor
            obsDim = [numUEs, 1]; % One throughput value per UE
            oInfo = rlNumericSpec(obsDim, 'LowerLimit', 0, 'UpperLimit', Inf);
            
            combinations = nchoosek(1:numUEs, usersPerTTI);

            % Step 2: Generate all permutations for each combination
            allActions = [];
            for i = 1:size(combinations, 1)
                permsOfCombination = perms(combinations(i, :)); % Get all orderings
                allActions = [allActions; permsOfCombination]; % Store results
            end

            aInfo = rlFiniteSetSpec(num2cell(allActions, 2));

            obj@rl.env.MATLABEnvironment(oInfo, aInfo);
            obj = obj@nrScheduler();

            obj.oInfo = oInfo;
            obj.aInfo = aInfo;
            
            obj.numUEs = numUEs;

            obj.UsersPerTTI = usersPerTTI;

            obj.UEOrder = 1 : obj.UsersPerTTI;

            % Define observation space (throughput for each UE)
            obj.Agent = createAgent(obj); 

            obj.UEThroughput = zeros(obj.numUEs, 1);      
        end

        function agent = createAgent(obj)
            
            agentOpts = rlDQNAgentOptions(...
                'UseDoubleDQN', true, ... % Helps stabilize learning
                'TargetSmoothFactor', 1e-3, ...
                'DiscountFactor', 0.99, ...
                'MiniBatchSize', 64, ...
                'ExperienceBufferLength', 1e6, ...
                'EpsilonGreedyExploration', rl.option.EpsilonGreedyExploration('Epsilon', 0.1) ...
            );

            agent = rlDQNAgent(obj.oInfo, obj.aInfo, agentOpts);
        end

        function transmissionCallback(obj, event)
            % Listener function: Trains RL agent and updates UE order
            ueThroughput = event.Throughput;

            % Get last action (UE order given previously)
            lastAction = {obj.UEOrder};
            
            % Get last observation (previous throughput per UE)
            lastObservation = {ueThroughput};  
            
            % Compute Jain's Fairness Index
            sumRate = sum(ueThroughput);
            sumRateSquared = sum(ueThroughput .^ 2);
            
            if sumRateSquared > 0
                JFI = (sumRate^2) / (obj.numUEs * sumRateSquared);
            else
                JFI = 0; % Avoid division by zero
            end

            alpha = 1;  % Weight for throughput
            beta = 1e6;   % Penalty weight for fairness deviation

            % Compute reward (e.g., sum of throughput values)
            reward = alpha * sumRate - beta * (1 - JFI) + randn;

            % Define next observation (use last observation as a placeholder)
            nextObservation = {ueThroughput + randn};  
        
            % Store experience in a struct
            experience.Observation = lastObservation;
            experience.Action = lastAction;
            experience.Reward = reward;
            experience.NextObservation = nextObservation;
            experience.IsDone = false; % Episode is not done

            obj.Agent.ExperienceBuffer.append(experience);

            trainOpts = rlTrainingOptions(...
                MaxEpisodes= 70, ...
                MaxStepsPerEpisode=100, ...        
                StopTrainingCriteria="AverageSteps", ...
                StopTrainingValue=100000, ...
                Plots="training-progress");
            
            % experience = {obj.UEOrder, ueThroughput};
            obj.Agent.train(obj, trainOpts);
        
            % Get next action from agent (new UE order)
            nextAction = getAction(obj.Agent, lastObservation);
            
            % Update UE order for next transmission
            obj.UEOrder = cell2mat(nextAction);
        end

        function initialState = reset(obj)
            initialState = obj.UEThroughput;
        end
        
        function [nextObservation, reward, isDone] = step(obj, action)
            obj.UEOrder = action;
            eventData = obj.EventData;
            [~, throughput] = dummyScheduleNewTransmissionsDL(eventData.Object, eventData.TimeResource, eventData.FrequencyResource, eventData.SchedulingInfo);
            nextObservation = {throughput + randn};

            sumRate = sum(throughput);
            sumRateSquared = sum(throughput .^ 2);
            
            if sumRateSquared > 0
                JFI = (sumRate^2) / (obj.numUEs * sumRateSquared);
            else
                JFI = 0; % Avoid division by zero
            end

            alpha = 1;  % Weight for throughput
            beta = 1e6;   % Penalty weight for fairness deviation

            % Compute reward (e.g., sum of throughput values)
            reward = alpha * sumRate - beta * (1 - JFI) + randn;
            isDone = false;
        end
    end

    methods (Access = protected)
        function dlAssignments = scheduleNewTransmissionsDL(obj, timeResource, frequencyResource, schedulingInfo)
            %scheduleNewTransmissionsDL Assign resources for new DL transmissions in a transmission time interval (TTI)
            %   DLGRANTS = scheduleNewTransmissionsDL(OBJ, TIMERESOURCE,
            %   FREQUENCYRESOURCE, SCHEDULINGINFO) assigns the time and
            %   frequency resources defined by TIMERESOURCE and
            %   FREQUENCYRESOURCE, respectively, to different UEs for new
            %   transmissions. The scheduler invokes this function after
            %   satisfying the retransmission requirements for UEs, if any. As a
            %   result, not all frequency resources of the bandwidth would
            %   be available for new transmission scheduling in this TTI.
            %
            %   TIMERESOURCE represents the TTI i.e., the time symbols
            %   which scheduler is scheduling. All the DL assignments
            %   generated as output would be over all these symbols. It is a
            %   structure with following fields:
            %       NFrame - Absolute frame number of the symbols getting scheduled
            %       NSlot - The slot number in the frame 'NFrame' whose symbols are getting scheduled
            %       SymbolAllocation - TTI symbol range as an array of two integers: [StartSym NumSym]
            %                     StartSym - Start symbol number (in the slot 'NSlot' of the frame 'NFrame') of the TTI getting scheduled
            %                     NumSym - Number of symbols getting scheduled
            %    All these fields values are 0-based.
            %
            %   FREQUENCYRESOURCE represents the frequency resources getting scheduled.
            %   If resource allocation type (See 'ResourceAllocationType' N-V parameter
            %   of nrGNB.configureScheduler) is 1 (i.e., RAT-1) then it is a bit array
            %   of length equal to number of RBs in the DL bandwidth. If resource
            %   allocation type is 0 (i.e., RAT-0) then it is a bit array of length
            %   equal to number of RBGs in the DL bandwidth. Value 0 at an index in the
            %   bit array means that the corresponding RB/RBG is available for
            %   scheduling, otherwise, it is considered unavailable for scheduling. If
            %   all the frequency resources in a TTI are consumed by retransmissions,
            %   then scheduler does not invoke this function for that TTI.
            %
            %   SCHEDULINGINFO is structure containing information to be used for
            %   scheduling. It has following fields:
            %   EligibleUEs - The set of eligible UEs for allocation in this TTI. It
            %   is an array of RNTI of the UEs. This set includes all the
            %   UEs connected to the gNB excluding the ones which match
            %   either of these criteria:
            %       (1) The UE is scheduled for retransmission in this TTI.
            %       (2) The UE does not have any queued data.
            %       (3) All the HARQ processes are blocked for the UE.
            %   If due to these criteria, none of the UEs qualify as eligible then
            %   scheduler does not invoke this function. MaxNumUsersTTI - Maximum
            %   number of UEs which can be scheduled in the TTI for new transmissions.
            %   The value is adjusted for the retransmissions scheduled in this TTI
            %   i.e., the value is count of UEs scheduled for retransmissions in this
            %   TTI subtracted from total maximum allowed users in a TTI (See
            %   MaxNumUsersPerTTI N-V parameter of nrGNB.configureScheduler). This
            %   function only gets called if value of this field is greater than zero.
            %
            %   Note that in addition to information in SCHEDULINGINFO, any other context
            %   in scheduler object, OBJ, can be used as information to decide resource allocation
            %   to UEs.
            %
            %   DLASSIGNMENTS is a struct array where each element represents DL assignment for a PDSCH, and
            %   has following information as fields:
            %   RNTI - Downlink grant is for this UE
            %   FrequencyAllocation - For RAT-0, a bit vector of length equal to number
            %                         of RBGs in the DL bandwidth. Value 1
            %                         at an index means that corresponding
            %                         RBG is assigned for the grant
            %                       - For RAT-1, a vector of two elements representing
            %                         start RB and number of RBs. Start RB is
            %                         0-indexed.
            %   W - Selected precoding matrix. It is an array of size
            %       NumLayers-by-P where P is the number of antenna
            %       ports. The number of rows in this matrix implicates the
            %       number of transmission layers to be used for PDSCH.
            %   MCSIndex - Selected modulation and coding scheme. This is
            %              a row index (0-based) in the table specified in nrGNB.MCSTable.

            % Read time resources
            scheduledSlot = timeResource.NSlot;
            startSym = timeResource.SymbolAllocation(1);
            numSym = timeResource.SymbolAllocation(2);

            % Read scheduling info structure
            eligibleUEs = schedulingInfo.EligibleUEs;
            numNewTxs = min(length(eligibleUEs), schedulingInfo.MaxNumUsersTTI);
            % Select index of the first UE for scheduling. After the last selected UE,
            % go in sequence and find index of the first eligible UE
            scheduledUEIndex = find(eligibleUEs>obj.LastSelectedUEDL, 1);
            if isempty(scheduledUEIndex)
                scheduledUEIndex = 1;
            end

            % Stores DL grants of the TTI
            dlAssignments = obj.DLGrantArrayStruct(1:numNewTxs);

            cellConfig = obj.CellConfig;
            schedulerConfig = obj.SchedulerConfig;
            ueContext = obj.UEContext;

            % Holds updated frequency occupancy status as the frequency resources are
            % keep getting allotted for new transmissions
            updatedFrequencyStatus = frequencyResource;

            % Select rank and precoding matrix for the eligible UEs
            numEligibleUEs = length(eligibleUEs);
            uePriority = ones(1,numEligibleUEs); % Higher the number higher is the priority
            W = cell(numEligibleUEs, 1); % To store selected precoding matrices for the UEs
            rank = zeros(numEligibleUEs, 1); % To store selected rank for the UEs
            rbRequirement = zeros(obj.NumUEs, 1); % To store RB requirement for UEs
            channelQuality = zeros(obj.NumUEs, cellConfig.NumResourceBlocks); % To store channel quality information for UEs
            cqiSizeArray = ones(cellConfig.NumResourceBlocks, 1);
            for i=1:numEligibleUEs
                rnti = eligibleUEs(i);
                eligibleUEContext = ueContext(rnti);
                csiMeasurement = eligibleUEContext.CSIMeasurementDL;
                csiMeasurementCQI = csiMeasurement.CSIRS.CQI*cqiSizeArray;
                channelQuality(rnti, :) = csiMeasurementCQI;
                if isempty(eligibleUEContext.CSIRSConfiguration)
                    numCSIRSPorts = obj.NumTransmitAntennas;
                else
                    numCSIRSPorts = eligibleUEContext.CSIRSConfiguration.NumCSIRSPorts;
                end
                [rank(i), W{i}] = selectRankAndPrecodingMatrixDL(obj, rnti, csiMeasurement, numCSIRSPorts);
                [bitsPerRB, rbRequirement(rnti)] = calculateRBRequirement(obj, rnti, obj.DLType, numSym, rank(i));
                ue_order = obj.UEOrder;
                if any(ue_order == i)
                    uePriority(i) = numEligibleUEs - find(ue_order == i) + 1;
                end
            end

            % Sort the UEs based on the calculated priority
            [~,ueIndices] = sort(uePriority,'descend');
            ueIndices = ueIndices(1:numNewTxs);

            updatedEligibleUEs = obj.randomizeUEsSelection(eligibleUEs, uePriority, ueIndices);

            % Rearrange indices for channel measurement, buffer status and RB requirement
            [~,~,matchingIndices] = intersect(updatedEligibleUEs,eligibleUEs,'stable');
            rank = rank(matchingIndices);
            W = W(matchingIndices);

            % Create the input structure for scheduling strategy
            schedulerInput = obj.SchedulerInputStruct;
            schedulerInput.linkDir = obj.DLType;
            if schedulerConfig.ResourceAllocationType % RAT-1
                % For MU-MIMO configuration
                schedulerInput.mcsRBG = zeros(1, numel(updatedEligibleUEs));
                schedulerInput.cqiRBG = channelQuality(updatedEligibleUEs,:);
                cqiSetRBG = floor(sum(schedulerInput.cqiRBG, 2)/size(schedulerInput.cqiRBG, 2));
                for i = 1:numel(updatedEligibleUEs)
                    schedulerInput.mcsRBG(i, 1) = selectMCSIndexDL(obj, cqiSetRBG(i), updatedEligibleUEs(i)); % MCS value
                end
            else % RAT-0
                rbRequirement = rbRequirement(eligibleUEs(matchingIndices));
                channelQuality = channelQuality(eligibleUEs(matchingIndices),:);
            end
            schedulerInput.eligibleUEs = updatedEligibleUEs;
            schedulerInput.selectedRank = rank;
            schedulerInput.bufferStatus = [ueContext(eligibleUEs(matchingIndices)).BufferStatusDL];
            schedulerInput.lastSelectedUE = obj.LastSelectedUEDL;
            schedulerInput.channelQuality = channelQuality;
            schedulerInput.freqOccupancyBitmap = updatedFrequencyStatus;
            schedulerInput.rbAllocationLimit = obj.RBAllocationLimitDL;
            schedulerInput.rbRequirement = rbRequirement;
            schedulerInput.maxNumUsersTTI = schedulingInfo.MaxNumUsersTTI;
            schedulerInput.numSym = numSym;

            % Implement scheduling strategy. Also ensure that the number of RBs
            % allotted to a UE in the slot does not exceed the limit as defined by the
            % class property 'RBAllocationLimit'
            % Run the scheduling strategy to select UEs, frequency resources and mcs indices
            if schedulerConfig.ResourceAllocationType % RAT-1
                [allottedUEs, freqAllocation, mcsIndex] = runSchedulingStrategyRAT1(obj, schedulerInput);
            else % RAT-0
                [allottedUEs, freqAllocation, mcsIndex] = runSchedulingStrategyRAT0(obj, schedulerInput);
            end

            numAllottedUEs = length(allottedUEs);
            for index = 1:numAllottedUEs
                selectedUE = allottedUEs(index);
                % Allot RBs to the selected UE in this TTI
                selectedUEIdx = find(updatedEligibleUEs == selectedUE, 1); % Find UE index in eligible UEs set
                % MCS offset value
                mcsOffset = fix(ueContext(selectedUE).MCSOffset(schedulerInput.linkDir+1));
                % Fill the new transmission RAT-1 downlink grant properties
                dlAssignments(index).RNTI = selectedUE;
                dlAssignments(index).FrequencyAllocation = freqAllocation(index, :);
                dlAssignments(index).MCSIndex = min(max(mcsIndex(index) - mcsOffset, 0), 27);
                dlAssignments(index).W = W{selectedUEIdx};

                % Mark frequency resources as assigned to the selected UE in this TTI
                if schedulerConfig.ResourceAllocationType % RAT-1
                    updatedFrequencyStatus(freqAllocation(index, 1)+1 : freqAllocation(index, 1)+freqAllocation(index, 2)) = 1;
                else % RAT-0
                    updatedFrequencyStatus = updatedFrequencyStatus | freqAllocation(index,:);
                end
            end

            dlAssignments = dlAssignments(1:numAllottedUEs); % Remove invalid trailing entries

            throughput = obj.UEThroughput;
            for i=1:numAllottedUEs
                selectedUE = allottedUEs(index);
                [bitsPerRB, rbRequirement(rnti)] = calculateRBRequirement(obj, dlAssignments(i).RNTI, obj.DLType, numSym, rank(selectedUEIdx));
                throughput(selectedUE) = throughput(selectedUE) + bitsPerRB * 1000/(numSym*cellConfig.SlotDuration/14);
            end

            eventData = ThroughputEventData(throughput, obj, timeResource, frequencyResource, schedulingInfo);
            obj.UEThroughput = throughput;
            obj.EventData = eventData;
            notify(obj, 'NewTransmissionRLScheduler', eventData);
        end

        function [dlAssignments, throughput] = dummyScheduleNewTransmissionsDL(obj, timeResource, frequencyResource, schedulingInfo)
            %scheduleNewTransmissionsDL Assign resources for new DL transmissions in a transmission time interval (TTI)
            %   DLGRANTS = scheduleNewTransmissionsDL(OBJ, TIMERESOURCE,
            %   FREQUENCYRESOURCE, SCHEDULINGINFO) assigns the time and
            %   frequency resources defined by TIMERESOURCE and
            %   FREQUENCYRESOURCE, respectively, to different UEs for new
            %   transmissions. The scheduler invokes this function after
            %   satisfying the retransmission requirements for UEs, if any. As a
            %   result, not all frequency resources of the bandwidth would
            %   be available for new transmission scheduling in this TTI.
            %
            %   TIMERESOURCE represents the TTI i.e., the time symbols
            %   which scheduler is scheduling. All the DL assignments
            %   generated as output would be over all these symbols. It is a
            %   structure with following fields:
            %       NFrame - Absolute frame number of the symbols getting scheduled
            %       NSlot - The slot number in the frame 'NFrame' whose symbols are getting scheduled
            %       SymbolAllocation - TTI symbol range as an array of two integers: [StartSym NumSym]
            %                     StartSym - Start symbol number (in the slot 'NSlot' of the frame 'NFrame') of the TTI getting scheduled
            %                     NumSym - Number of symbols getting scheduled
            %    All these fields values are 0-based.
            %
            %   FREQUENCYRESOURCE represents the frequency resources getting scheduled.
            %   If resource allocation type (See 'ResourceAllocationType' N-V parameter
            %   of nrGNB.configureScheduler) is 1 (i.e., RAT-1) then it is a bit array
            %   of length equal to number of RBs in the DL bandwidth. If resource
            %   allocation type is 0 (i.e., RAT-0) then it is a bit array of length
            %   equal to number of RBGs in the DL bandwidth. Value 0 at an index in the
            %   bit array means that the corresponding RB/RBG is available for
            %   scheduling, otherwise, it is considered unavailable for scheduling. If
            %   all the frequency resources in a TTI are consumed by retransmissions,
            %   then scheduler does not invoke this function for that TTI.
            %
            %   SCHEDULINGINFO is structure containing information to be used for
            %   scheduling. It has following fields:
            %   EligibleUEs - The set of eligible UEs for allocation in this TTI. It
            %   is an array of RNTI of the UEs. This set includes all the
            %   UEs connected to the gNB excluding the ones which match
            %   either of these criteria:
            %       (1) The UE is scheduled for retransmission in this TTI.
            %       (2) The UE does not have any queued data.
            %       (3) All the HARQ processes are blocked for the UE.
            %   If due to these criteria, none of the UEs qualify as eligible then
            %   scheduler does not invoke this function. MaxNumUsersTTI - Maximum
            %   number of UEs which can be scheduled in the TTI for new transmissions.
            %   The value is adjusted for the retransmissions scheduled in this TTI
            %   i.e., the value is count of UEs scheduled for retransmissions in this
            %   TTI subtracted from total maximum allowed users in a TTI (See
            %   MaxNumUsersPerTTI N-V parameter of nrGNB.configureScheduler). This
            %   function only gets called if value of this field is greater than zero.
            %
            %   Note that in addition to information in SCHEDULINGINFO, any other context
            %   in scheduler object, OBJ, can be used as information to decide resource allocation
            %   to UEs.
            %
            %   DLASSIGNMENTS is a struct array where each element represents DL assignment for a PDSCH, and
            %   has following information as fields:
            %   RNTI - Downlink grant is for this UE
            %   FrequencyAllocation - For RAT-0, a bit vector of length equal to number
            %                         of RBGs in the DL bandwidth. Value 1
            %                         at an index means that corresponding
            %                         RBG is assigned for the grant
            %                       - For RAT-1, a vector of two elements representing
            %                         start RB and number of RBs. Start RB is
            %                         0-indexed.
            %   W - Selected precoding matrix. It is an array of size
            %       NumLayers-by-P where P is the number of antenna
            %       ports. The number of rows in this matrix implicates the
            %       number of transmission layers to be used for PDSCH.
            %   MCSIndex - Selected modulation and coding scheme. This is
            %              a row index (0-based) in the table specified in nrGNB.MCSTable.

            % Read time resources
            scheduledSlot = timeResource.NSlot;
            startSym = timeResource.SymbolAllocation(1);
            numSym = timeResource.SymbolAllocation(2);

            % Read scheduling info structure
            eligibleUEs = schedulingInfo.EligibleUEs;
            numNewTxs = min(length(eligibleUEs), schedulingInfo.MaxNumUsersTTI);
            % Select index of the first UE for scheduling. After the last selected UE,
            % go in sequence and find index of the first eligible UE
            scheduledUEIndex = find(eligibleUEs>obj.LastSelectedUEDL, 1);
            if isempty(scheduledUEIndex)
                scheduledUEIndex = 1;
            end

            % Stores DL grants of the TTI
            dlAssignments = obj.DLGrantArrayStruct(1:numNewTxs);

            cellConfig = obj.CellConfig;
            schedulerConfig = obj.SchedulerConfig;
            ueContext = obj.UEContext;

            % Holds updated frequency occupancy status as the frequency resources are
            % keep getting allotted for new transmissions
            updatedFrequencyStatus = frequencyResource;

            % Select rank and precoding matrix for the eligible UEs
            numEligibleUEs = length(eligibleUEs);
            uePriority = ones(1,numEligibleUEs); % Higher the number higher is the priority
            W = cell(numEligibleUEs, 1); % To store selected precoding matrices for the UEs
            rank = zeros(numEligibleUEs, 1); % To store selected rank for the UEs
            rbRequirement = zeros(obj.NumUEs, 1); % To store RB requirement for UEs
            channelQuality = zeros(obj.NumUEs, cellConfig.NumResourceBlocks); % To store channel quality information for UEs
            cqiSizeArray = ones(cellConfig.NumResourceBlocks, 1);
            for i=1:numEligibleUEs
                rnti = eligibleUEs(i);
                eligibleUEContext = ueContext(rnti);
                csiMeasurement = eligibleUEContext.CSIMeasurementDL;
                csiMeasurementCQI = csiMeasurement.CSIRS.CQI*cqiSizeArray;
                channelQuality(rnti, :) = csiMeasurementCQI;
                if isempty(eligibleUEContext.CSIRSConfiguration)
                    numCSIRSPorts = obj.NumTransmitAntennas;
                else
                    numCSIRSPorts = eligibleUEContext.CSIRSConfiguration.NumCSIRSPorts;
                end
                [rank(i), W{i}] = selectRankAndPrecodingMatrixDL(obj, rnti, csiMeasurement, numCSIRSPorts);
                [bitsPerRB, rbRequirement(rnti)] = calculateRBRequirement(obj, rnti, obj.DLType, numSym, rank(i));
                ue_order = obj.UEOrder;
                if any(ue_order == i)
                    uePriority(i) = numEligibleUEs - find(ue_order == i) + 1;
                end
            end

            % Sort the UEs based on the calculated priority
            [~,ueIndices] = sort(uePriority,'descend');
            ueIndices = ueIndices(1:numNewTxs);

            updatedEligibleUEs = obj.randomizeUEsSelection(eligibleUEs, uePriority, ueIndices);

            % Rearrange indices for channel measurement, buffer status and RB requirement
            [~,~,matchingIndices] = intersect(updatedEligibleUEs,eligibleUEs,'stable');
            rank = rank(matchingIndices);
            W = W(matchingIndices);

            % Create the input structure for scheduling strategy
            schedulerInput = obj.SchedulerInputStruct;
            schedulerInput.linkDir = obj.DLType;
            if schedulerConfig.ResourceAllocationType % RAT-1
                % For MU-MIMO configuration
                schedulerInput.mcsRBG = zeros(1, numel(updatedEligibleUEs));
                schedulerInput.cqiRBG = channelQuality(updatedEligibleUEs,:);
                cqiSetRBG = floor(sum(schedulerInput.cqiRBG, 2)/size(schedulerInput.cqiRBG, 2));
                for i = 1:numel(updatedEligibleUEs)
                    schedulerInput.mcsRBG(i, 1) = selectMCSIndexDL(obj, cqiSetRBG(i), updatedEligibleUEs(i)); % MCS value
                end
            else % RAT-0
                rbRequirement = rbRequirement(eligibleUEs(matchingIndices));
                channelQuality = channelQuality(eligibleUEs(matchingIndices),:);
            end
            schedulerInput.eligibleUEs = updatedEligibleUEs;
            schedulerInput.selectedRank = rank;
            schedulerInput.bufferStatus = [ueContext(eligibleUEs(matchingIndices)).BufferStatusDL];
            schedulerInput.lastSelectedUE = obj.LastSelectedUEDL;
            schedulerInput.channelQuality = channelQuality;
            schedulerInput.freqOccupancyBitmap = updatedFrequencyStatus;
            schedulerInput.rbAllocationLimit = obj.RBAllocationLimitDL;
            schedulerInput.rbRequirement = rbRequirement;
            schedulerInput.maxNumUsersTTI = schedulingInfo.MaxNumUsersTTI;
            schedulerInput.numSym = numSym;

            % Implement scheduling strategy. Also ensure that the number of RBs
            % allotted to a UE in the slot does not exceed the limit as defined by the
            % class property 'RBAllocationLimit'
            % Run the scheduling strategy to select UEs, frequency resources and mcs indices
            if schedulerConfig.ResourceAllocationType % RAT-1
                [allottedUEs, freqAllocation, mcsIndex] = runSchedulingStrategyRAT1(obj, schedulerInput);
            else % RAT-0
                [allottedUEs, freqAllocation, mcsIndex] = runSchedulingStrategyRAT0(obj, schedulerInput);
            end

            numAllottedUEs = length(allottedUEs);
            for index = 1:numAllottedUEs
                selectedUE = allottedUEs(index);
                % Allot RBs to the selected UE in this TTI
                selectedUEIdx = find(updatedEligibleUEs == selectedUE, 1); % Find UE index in eligible UEs set
                % MCS offset value
                mcsOffset = fix(ueContext(selectedUE).MCSOffset(schedulerInput.linkDir+1));
                % Fill the new transmission RAT-1 downlink grant properties
                dlAssignments(index).RNTI = selectedUE;
                dlAssignments(index).FrequencyAllocation = freqAllocation(index, :);
                dlAssignments(index).MCSIndex = min(max(mcsIndex(index) - mcsOffset, 0), 27);
                dlAssignments(index).W = W{selectedUEIdx};

                % Mark frequency resources as assigned to the selected UE in this TTI
                if schedulerConfig.ResourceAllocationType % RAT-1
                    updatedFrequencyStatus(freqAllocation(index, 1)+1 : freqAllocation(index, 1)+freqAllocation(index, 2)) = 1;
                else % RAT-0
                    updatedFrequencyStatus = updatedFrequencyStatus | freqAllocation(index,:);
                end
            end

            dlAssignments = dlAssignments(1:numAllottedUEs); % Remove invalid trailing entries

            throughput = obj.UEThroughput;
            for i=1:numAllottedUEs
                selectedUE = allottedUEs(index);
                [bitsPerRB, rbRequirement(rnti)] = calculateRBRequirement(obj, dlAssignments(i).RNTI, obj.DLType, numSym, rank(selectedUEIdx));
                throughput(selectedUE) = throughput(selectedUE) + bitsPerRB * 1000/(numSym*cellConfig.SlotDuration/14);
            end
        end
    end

    methods (Access = protected, Hidden)
        function [reTxUEs, updatedFrequencyAllocation, dlGrants] = scheduleRetransmissionsDL(obj, timeResource)
            %scheduleRetransmissionsDL Assign resources of a set of contiguous DL symbols representing a TTI, of the specified slot for downlink retransmissions
            % Return the downlink assignments to the UEs which are allotted
            % retransmission opportunity and the updated frequency-occupancy-status to
            % convey what all frequency resources are used. All UEs are checked if they
            % require retransmission for any of their HARQ processes. If there are
            % multiple such HARQ processes for a UE then one HARQ process is selected
            % randomly among those. All UEs get maximum 1 retransmission opportunity in
            % a TTI

            schedulerConfig = obj.SchedulerConfig;
            if schedulerConfig.ResourceAllocationType % RAT-1
                % Holds updated RB occupancy status as the RBs keep getting allotted for
                % retransmissions
                updatedFrequencyAllocation = zeros(1, obj.CellConfig.NumResourceBlocks);
            else % RAT-0
                % Holds updated RBG occupancy status as the RBGs keep getting allotted for
                % retransmissions
                updatedFrequencyAllocation = zeros(1, obj.UEContext(1).NumRBGs);
            end

            % Read information about time resource scheduled in this TTI
            scheduledSlot = timeResource.NSlot;
            startSym = timeResource.SymbolAllocation(1);
            numSym = timeResource.SymbolAllocation(2);

            reTxGrantCount = 0;
            isAssigned=0;
            numUEs = obj.NumUEs;
            % Store UEs which get retransmission opportunity
            reTxUEs = zeros(numUEs, 1);
            % Store retransmission DL grants of this TTI
            dlGrants = repmat(obj.DLGrantInfoStruct, numUEs, 1);

            % Create a random permutation of UE RNTIs, to define the order in which
            % retransmission assignments would be done for this TTI
            reTxAssignmentOrder = randperm(numUEs);

            % Calculate offset of currently scheduled slot from the current slot
            slotOffset = scheduledSlot - obj.CurrSlot;
            if scheduledSlot < obj.CurrSlot
                slotOffset = slotOffset + obj.CellConfig.NumSlotsFrame; % Scheduled slot is in next frame
            end

            % Consider retransmission requirement of the UEs as per
            % reTxAssignmentOrder
            for i = 1:length(reTxAssignmentOrder) % For each UE
                % Stop assigning resources if the allocations are done for maximum users
                if reTxGrantCount >= schedulerConfig.MaxNumUsersPerTTI
                    break;
                end
                selectedUE = reTxAssignmentOrder(i);
                ueContext = obj.UEContext(selectedUE);
                reTxContextUE = ueContext.RetransmissionContextDL;
                failedRxHarqs = find(reTxContextUE==1);
                if ~isempty(failedRxHarqs)
                    % Select one HARQ process randomly
                    selectedHarqId = failedRxHarqs(randi(length(failedRxHarqs)))-1;
                    % Read TBS. Retransmission grant TBS also needs to be big enough to
                    % accommodate the packet
                    lastGrant = ueContext.HarqStatusDL{selectedHarqId+1};
                    % Select rank and precoding matrix as per the last transmission
                    rank = lastGrant.NumLayers;
                    W = lastGrant.W;

                    % Non-adaptive retransmissions
                    if schedulerConfig.ResourceAllocationType % RAT-1
                        lastGrantNumSym = lastGrant.NumSymbols;
                        lastGrantNumRBs = lastGrant.FrequencyAllocation(2);
                        % Ensure that total REs are at least equal to REs in original grant
                        numResourceBlocks = ceil(lastGrantNumSym*lastGrantNumRBs/numSym);
                        startRBIndex = find(updatedFrequencyAllocation == 0, 1)-1;
                        if numResourceBlocks <= (obj.CellConfig.NumResourceBlocks-startRBIndex)
                            % Retransmission TBS requirement have met
                            isAssigned = 1;
                            frequencyAllocation = [startRBIndex numResourceBlocks];
                            mcs = lastGrant.MCSIndex;
                            % Mark the allotted resources as occupied
                            updatedFrequencyAllocation(startRBIndex+1:startRBIndex+numResourceBlocks) = 1;
                        end
                    else % RAT-0
                        % Assign resources and MCS for retransmission
                        [isAssigned, frequencyAllocation, mcs] = getRetxResourcesNonAdaptive(obj, selectedUE, ...
                            updatedFrequencyAllocation, numSym, lastGrant);
                        if isAssigned % Mark the allotted resources as occupied
                            updatedFrequencyAllocation = updatedFrequencyAllocation | frequencyAllocation;
                        end
                    end

                    if isAssigned
                        % Fill the retransmission downlink grant properties
                        grant = obj.DLGrantInfoStruct;
                        grant.RNTI = selectedUE;
                        grant.Type = 'reTx';
                        grant.HARQID = selectedHarqId;
                        grant.ResourceAllocationType = schedulerConfig.ResourceAllocationType;
                        grant.FrequencyAllocation = frequencyAllocation;
                        grant.StartSymbol = startSym;
                        grant.NumSymbols = numSym;
                        grant.SlotOffset = slotOffset;
                        grant.MCSIndex = mcs;
                        grant.NDI = ueContext.HarqNDIDL(selectedHarqId+1); % Fill same NDI (for retransmission)
                        grant.FeedbackSlotOffset = getPDSCHFeedbackSlotOffset(obj, slotOffset);
                        grant.DMRSLength = obj.PDSCHDMRSLength;
                        grant.MappingType = schedulerConfig.PDSCHMappingType;
                        grant.NumLayers = rank;
                        grant.W = W;
                        grant.NumCDMGroupsWithoutData = 2; % Number of CDM groups without data (1...3)
                        grant.BeamIndex = [];

                        % Set the RV
                        harqProcessContext = ueContext.HarqProcessesDL(selectedHarqId+1);
                        harqProcess = nr5g.internal.nrUpdateHARQProcess(harqProcessContext, 1);
                        grant.RV = harqProcess.RVSequence(harqProcess.RVIdx(1));

                        reTxGrantCount = reTxGrantCount+1;
                        reTxUEs(reTxGrantCount) = selectedUE;
                        dlGrants(reTxGrantCount) = grant;
                        isAssigned = 0;
                    end
                end
            end
            reTxUEs = reTxUEs(1:reTxGrantCount);
            dlGrants = dlGrants(1:reTxGrantCount); % Remove all empty elements

            % Initialize throughput if not already done
            throughput = obj.UEThroughput;
            
            % Iterate over retransmitting UEs and update throughput
            for i = 1:reTxGrantCount
                selectedUE = reTxUEs(i);
                grant = dlGrants(i);
                
                % Calculate RB requirement
                [bitsPerRB, ~] = calculateRBRequirement(obj, selectedUE, obj.DLType, numSym, grant.NumLayers);
                
                % Update throughput
                throughput(selectedUE) = throughput(selectedUE) + bitsPerRB * 1000 / (numSym * obj.CellConfig.SlotDuration / 14);
            end
            
            % Store the updated throughput
            obj.UEThroughput = throughput;
        end
    end
end
