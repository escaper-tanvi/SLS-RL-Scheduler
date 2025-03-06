% Compare the wideband SINR for CSI-RS and SRS based DL measurements in TDD
% system.
%
% Copyright 2024 The MathWorks, Inc.

wirelessnetworkSupportPackageCheck
numUEs = 200; % Total number of UEs in the scenario

% Run the scenario to estimate DL CSI using SRS
srsDLSINR = runScenarios("SRS", numUEs);

% Run the scenario to estimate DL CSI using CSI-RS
csirsDLSINR = runScenarios("CSI-RS", numUEs);

% Extracing PDSCH data from the scheduled slots
csirsDLSINRNonZero = csirsDLSINR(csirsDLSINR~=0);
srsDLSINRNonZero = srsDLSINR(srsDLSINR~=0);

data = {srsDLSINRNonZero(:), csirsDLSINRNonZero(:)};

% Plot the DL SINR CDF for different reference signals
legendName = ["SRS" "CSI-RS"];
xLabel = "SINR (dB)";
figureTitle = "CDF of DL SINR for all UEs";
calculateAndPlotSINRCDF(data, legendName, figureTitle, xLabel);

%% Local functions
function dlSINR = runScenarios(referenceSignal, numUEs)
% The function simulates the scenario for the comparison of the SINRs
    clear displayEventdata; % To clear the persistent variable

    % Create a wireless network simulator.
    rng("default")           % Reset the random number generator
    numFrameSimulation = 50; % Simulation time in terms of number of 10 ms frames
    networkSimulator = wirelessNetworkSimulator.init;

    % Set phyAbstractionType to 'linkToSystemMapping' to support SRS based DL
    % SU-MIMO
    phyAbstractionType = "linkToSystemMapping";

    % Create a gNB. Specify the position, the carrier frequency, the channel bandwidth,
    % the subcarrier spacing, the number of transmit antennas, the number of receive
    % antennas, and the receive gain of the node.
    gNB = nrGNB(Position=[0 0 30],CarrierFrequency=2.6e9,ChannelBandwidth=5e6,SubcarrierSpacing=15e3,DuplexMode="TDD",...
        NumTransmitAntennas=16,NumReceiveAntennas=16,ReceiveGain=11,PHYAbstractionMethod=phyAbstractionType, NumResourceBlocks=24);

    % Configure the scheduler to use SRS or CSI-RS based downlink measurement
    configureScheduler(gNB, CSIMeasurementSignalDL=referenceSignal,MaxNumUsersPerTTI=25)

    % Create 500 UE nodes.
    ueRelPosition = [(rand(numUEs,1)-0.5)*10000 (rand(numUEs,1)-0.5)*10000 zeros(numUEs,1)];
    % Convert spherical to Cartesian coordinates considering gNB position as origin
    [xPos,yPos,zPos] = sph2cart(deg2rad(ueRelPosition(:,2)),deg2rad(ueRelPosition(:,3)), ...
        ueRelPosition(:,1));

    % Convert to absolute Cartesian coordinates
    uePositions = [xPos yPos zPos] + gNB.Position;
    ueNames = "UE-" + (1:size(uePositions,1));

    % Specify the name, the position, the number of transmit
    % antennas, the number of receive antennas, and the receive gain of each UE node.
    UEs = nrUE(Name=ueNames,Position=uePositions,NumTransmitAntennas=4,NumReceiveAntennas=4,ReceiveGain=0,PHYAbstractionMethod=phyAbstractionType);

    rlcBearer = nrRLCBearerConfig(SNFieldLength=6,BucketSizeDuration=10); % Create an RLC bearer configuration object.
    connectUE(gNB,UEs,RLCBearerConfig=rlcBearer)                          % Establish an RLC bearer between the gNB and UE nodes

    appDataRate = 40e3; % Application data rate in kilobits per second (kbps)
    for ueIdx = 1:length(UEs)
        % Install the DL application traffic on gNB for the UE node
        dlApp = networkTrafficOnOff(GeneratePacket=true,OnTime=numFrameSimulation*10e-3,...
            OffTime=0,DataRate=appDataRate);
        addTrafficSource(gNB,dlApp,DestinationNode=UEs(ueIdx))

        % Install the UL application traffic on the UE node for the gNB
        ulApp = networkTrafficOnOff(GeneratePacket=true,OnTime=numFrameSimulation*10e-3,...
            OffTime=0,DataRate=appDataRate);
        addTrafficSource(UEs(ueIdx),ulApp)
    end

    % Add the gNB and UE nodes to the network simulator.
    addNodes(networkSimulator,gNB)
    addNodes(networkSimulator,UEs)

    % Create an N-by-N array of link-level channels, where N represents the number of
    % nodes in the cell. An element at index (i,j) contains the channel instance
    % from node i to node j. If the element at index (i,j) is empty, it indicates
    % the absence of a channel from node i to node j. Here i and j represents the
    % node IDs.
    channelConfig = struct(DelayProfile="CDL-D", DelaySpread=30e-9);
    channels = createCDLChannels(channelConfig,gNB,UEs);

    % Create a custom channel model using channels and install the custom
    % channel on the simulator. Network simulator applies the channel to a
    % packet in transit before passing it to the receiver.
    channel = hNRCustomChannelModel(channels,struct(PHYAbstractionMethod=phyAbstractionType));
    addChannelModel(networkSimulator,@channel.applyChannelModel)

    % Run the simulation for the specified numFrameSimulation frames.
    % Calculate the simulation duration (in seconds)
    simulationTime = numFrameSimulation * 1e-2;

    addlistener(UEs, 'PacketReceptionEnded', @(src, eventData) displayEventdata(src, eventData, numUEs));
    % Run the simulation
    run(networkSimulator,simulationTime);

    % extract the DL SINR for the reference signal
    dlSINR = getappdata(0, 'dlsinr');
end

% Set up CDL channel instances for the cell. For DL, set up a CDL channel instance 
% from the gNB to each UE node. For UL, set up a CDL channel instance from each 
% UE node to the gNB.
function channels = createCDLChannels(channelConfig,gNB,UEs)
%createCDLChannels Create channels between gNB and UEs in a cell
%   CHANNELS = createCDLChannels(CHANNELCONFIG,GNB,UES) creates channels
%   between the GNB and UE nodes in a cell.
%
%   CHANNELS is an N-by-N array, where N represents the number of nodes in the cell.
%
%   CHANNLECONFIG is a structure with these fields - DelayProfile and
%   DelaySpread.
%
%   GNB is an nrGNB object.
%
%   UES is an array of nrUE objects.

numUEs = length(UEs);
numNodes = length(gNB) + numUEs;
% Create channel matrix to hold the channel objects
channels = cell(numNodes,numNodes);

% Obtain the sample rate of waveform
waveformInfo = nrOFDMInfo(gNB.NumResourceBlocks,gNB.SubcarrierSpacing/1e3);
sampleRate = waveformInfo.SampleRate;
channelFiltering = strcmp(gNB.PHYAbstractionMethod,'none');

for ueIdx = 1:numUEs
    % Configure the uplink channel model between the gNB and UE nodes
    channel = nrCDLChannel;
    channel.DelayProfile = channelConfig.DelayProfile;
    channel.DelaySpread = channelConfig.DelaySpread;
    channel.Seed = 730 + (ueIdx - 1);
    channel.CarrierFrequency = gNB.CarrierFrequency;
    channel = hArrayGeometry(channel, UEs(ueIdx).NumTransmitAntennas,gNB.NumReceiveAntennas,...
        "uplink");
    channel.SampleRate = sampleRate;
    channel.ChannelFiltering = channelFiltering;
    channel.TransmitAntennaArray.Element = 'isotropic';
    channels{UEs(ueIdx).ID, gNB.ID} = channel;

    % Configure the downlink channel model between the gNB and UE nodes.
    % Channel reciprocity in case of TDD
    cdlUL = clone(channel);
    cdlUL.swapTransmitAndReceive();
    channels{gNB.ID, UEs(ueIdx).ID} = cdlUL;
end
end

% Extract DL SINR for PDSCH slots
function displayEventdata(~, event, numUEs)
    persistent dlSINR ueCount;
    if isempty(dlSINR)
        dlSINR = [];
        ueCount = zeros(numUEs, 1);
    end
    if strcmp(event.EventName, "PacketReceptionEnded") && strcmp(event.Data.SignalType, "PDSCH") && event.Data.CurrentTime*1e3 > 70
        % Extracting SINRS for values for slots greater than 70 as SRS transmissions will be
        % completed for all UEs. Else it will use default rank, precoder and MCS.
        ueCount(event.Data.RNTI, 1) = ueCount(event.Data.RNTI, 1) + 1;
        dlSINR(event.Data.RNTI, ueCount(event.Data.RNTI, 1)) = event.Data.SINR;
        setappdata(0, 'dlsinr', dlSINR);
    end
end

function calculateAndPlotSINRCDF(data, legendName, figureTitle, xLabel)

    fig = figure;
    ax = axes('Parent',fig);
    numPlots = 2;
    for i = 1:numPlots
        % Calculate the empirical cumulative distribution function F, evaluated at
        % x, using the data
        [x, F] = stairs(sort(data{i}),(1:length(data{i}))/length(data{i}));
        % Include a starting value, required for accurate plot
        x = [x(1); x];
        F = [0; F];

        % Plot the estimated empirical cdf
        plot(ax,x,F,LineWidth=2);
        hold(ax,"on");
        grid on;
    end
    hold(ax,"off");
    legend(ax,legendName,'Location','best');
    xlabel(ax, xLabel);
    ylabel(ax, 'C.D.F');
    title(ax, figureTitle)
end