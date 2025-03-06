classdef helperNRGridVisualizer < handle
    %helperNRGridVisualizer Scheduler log visualization
    %   The class implements visualization of logs by querying from the
    %   logger (helperNRSchedulingLogger object).
    %   The following two types of visualizations are shown:
    %    (i) Display of CQI values for UEs over the bandwidth
    %   (ii) Display of resource grid assignment to UEs. This 2D time-frequency
    %        grid shows the RB allocation to the UEs in the previous slot for
    %        symbol based scheduling and previous frame for slot based
    %        scheduling. HARQ process for the assignments is also shown
    %        alongside the UE's RNTI
    %
    %   helperNRGridVisualizer methods:
    %
    %   plotRBGrids         - Plot RB grid visualization
    %   plotCQIRBGrids      - Plot the CQI grid visualization
    %
    %   helperNRGridVisualizer Name-Value pairs:
    %
    %   CellOfInterest    - Cell ID to which the visualization object belongs
    %   SchedulingLogger  - Scheduling logger handle object
    %   LinkDirection     - Flag to indicate the plots to visualize

    %   Copyright 2023-2024 The MathWorks, Inc.

    properties
        %CellOfInterest Cell ID to which the visualization object belongs
        CellOfInterest (1, 1) {mustBeInteger, mustBeInRange(CellOfInterest, 0, 1007)} = 1;

        %SchedulingLogger MAC logger handle object
        SchedulingLogger

        %LinkDirection  Indicates the plots to visualize
        % It takes the values 0, 1, 2 and represent downlink, uplink, and both
        % respectively. Default value is 2.
        LinkDirection (1, 1) {mustBeInteger, mustBeInRange(LinkDirection, 0, 2)} = 2;
    end

    properties(Hidden)
        %ResourceGridVisualization Switch to turn on/off the resource grid visualization (resource-grid occupancy)
        ResourceGridVisualization = false;

        %RGMaxRBsToDisplay Max number of RBs displayed in resource grid visualization
        RGMaxRBsToDisplay = 20

        %RGMaxSlotsToDisplay Max number of slots displayed in resource grid visualization
        RGMaxSlotsToDisplay = 10

        %CQIGridVisualization Switch to turn on/off the CQI grid visualization
        CQIGridVisualization = false;

        %CVMaxRBsToDisplay Max number of RBs to be displayed in CQI visualization
        CVMaxRBsToDisplay = 20

        %CVMaxUEsToDisplay Max number of UEs to be displayed in CQI visualization
        CVMaxUEsToDisplay = 10

        %CQIVisualizationFigHandle Handle of the CQI visualization
        CQIVisualizationFigHandle

        %RGVisualizationFigHandle Handle of the resource grid visualization
        RGVisualizationFigHandle

        %IsLogReplay Flag to decide the type of post-simulation visualization
        % whether to show plain replay of the resource assignment during
        % simulation or of the selected slot (or frame). During the
        % post-simulation visualization, setting the value to 1 just
        % replays the resource assignment of the simulation frame-by-frame
        % (or slot-by-slot). Setting value to 0 gives the option to select
        % a particular frame (or slot) to see the way resources are
        % assigned in the chosen frame (or slot)
        IsLogReplay

        %SimulationLogs Simulation logs of the network
        SimulationLogs
    end

    properties (Constant)
        %NumSym Number of symbols in a slot
        NumSym = 14;

        % Duplexing mode related constants
        %FDDDuplexMode Frequency division duplexing mode
        FDDDuplexMode = 0;
        %TDDDuplexMode Time division duplexing mode
        TDDDuplexMode = 1;

        % Constants related to scheduling type
        %SymbolBased Symbol based scheduling
        SymbolBased = 1;
        %SlotBased Slot based scheduling
        SlotBased = 0;

        % Constants related to downlink and uplink information. These
        % constants are used for indexing logs and identifying plots
        %DownlinkIdx Index for all downlink information
        DownlinkIdx = 1;
        %UplinkIdx Index for all downlink information
        UplinkIdx = 2;

        %ColorCoding Mapping of a range of CQI values to particular color
        ColorCoding = [0.85 0.32 0.09 ; 0.85 0.32 0.09; 0.88 0.50 0.09; 0.88 0.50 0.09; ...
            0.93 0.69 0.13; 0.93 0.69 0.13; 0.98 0.75 0.26; 0.98 0.75 0.26; ...
            0.98 0.82 0.14; 0.98 0.82 0.14; 0.8 0.81 0.16; 0.8 0.81 0.16; ...
            0.68 0.71 0.18; 0.68 0.71 0.18; 0.46 0.67 0.18; 0.46 0.67 0.18];

        %MaxCells Maximum number of cells
        MaxCells = 1008;
    end

    properties (SetAccess=public)
        %NumUEs Count of UEs
        NumUEs

        %NumHARQ Number of HARQ processes
        % The default value is 16 HARQ processes
        NumHARQ (1, 1) {mustBeInteger, mustBeInRange(NumHARQ, 1, 16)} = 16;

        %NumFrames Number of frames in simulation
        NumFrames

        %SchedulingType Type of scheduling (slot based or symbol based)
        % Value 0 means slot based and value 1 means symbol based. The
        % default value is 0
        SchedulingType (1, 1) {mustBeMember(SchedulingType, [0, 1])} = 0;

        %DuplexMode Duplexing mode
        % Frequency division duplexing (FDD) or time division duplexing (TDD)
        % Value 0 means FDD and 1 means TDD. The default value is 0
        DuplexMode (1, 1) {mustBeMember(DuplexMode, [0, 1])} = 0;

        %ColumnIndexMap Mapping the column names of logs to respective column indices
        % It is a map object
        ColumnIndexMap

        %NumRBs Number of resource blocks
        % A vector of two elements. First element represents number of
        % PDSCH RBs and second element represents number of PUSCH RBs
        NumRBs = zeros(2, 1);

        %NumSlotsFrame Number of slots in 10ms time frame
        NumSlotsFrame

        %CurrFrame Current frame number
        CurrFrame

        %CurrSlot Current slot in the frame
        CurrSlot

        %NumLogs Number of logs to be created based on number of links
        NumLogs = 2;

        %SymSlotInfo Information about how each symbol/slot (UL/DL/Guard) is allocated
        SymSlotInfo

        %PlotIds IDs of the plots
        PlotIds

        %RBItemsList Items List for RBs drop down for DL and UL
        RBItemsList = cell(2, 1);

        %Resource grid information related properties
        % ResourceGrid In FDD mode first element contains downlink resource
        % grid allocation status and second element contains uplink
        % resource grid allocation status. In TDD mode first element
        % contains resource grid allocation status for downlink and uplink.
        % Each element is a 2D resource grid of N-by-P matrix where 'N' is
        % the number of slot or symbols and 'P' is the number of RBs in the
        % bandwidth to store how UEs are assigned different time-frequency
        % resources.
        ResourceGrid = cell(2, 1);

        %ResourceGridReTxInfo First element contains transmission status
        % in downlink and second element contains transmission status in
        % uplink for FDD mode. In TDD mode first element contains
        % transmission status for both downlink and uplink. Each element is
        % a 2D resource grid of N-by-P matrix where 'N' is the number of
        % slot or symbols and 'P' is the number of RBs in the bandwidth to
        % store type:new-transmission or retransmission.
        ResourceGridReTxInfo = cell(2, 1);

        %ResourceGridHarqInfo In FDD mode first element contains downlink
        % HARQ information and second element contains uplink HARQ
        % information. In TDD mode first element contains HARQ information
        % for downlink and uplink. Each element is a 2D resource grid of
        % N-by-P matrix where 'N' is the number of slot or symbols and 'P'
        % is the number of RBs in the bandwidth to store the HARQ process
        ResourceGridHarqInfo

        %ResourceGridTextHandles Text handles to display of the RNTI for the RBs
        ResourceGridTextHandles

        %ResourceGridInfo Text information for ResourceGridTextHandles.
        % First element contains text related to downlink and second
        % element contains text related to uplink for FDD mode. In TDD mode
        % first element contains text related to both downlink and uplink.
        ResourceGridInfo = cell(2, 1)

        %RVCurrView Type of scheduler scheduling information displayed in
        % CQI Visualization. Value 1 represents downlink and value 2
        % represents uplink
        RVCurrView = 1

        %RGTxtHandle UI control handle to display the frame number in resource grid visualization
        RGTxtHandle

        %RGSlotTxtHandle UI control handle to display the slot number in resource grid visualization
        RGSlotTxtHandle

        %RGLowerRBIndex Index of the first RB displayed in resource grid visualization
        RGLowerRBIndex = 0

        %RGUpperRBIndex Index of the last RB displayed in resource grid visualization
        RGUpperRBIndex

        %RGLowerSlotIndex Index of the first slot displayed in resource grid visualization
        RGLowerSlotIndex = 0

        %RGUpperSlotIndex Index of the last slot displayed in resource grid visualization
        RGUpperSlotIndex

        % CQI information related properties
        %CQIInfo It contains downlink CQI informatiom. Each element is
        % a N-by-P matrix where 'N' is the number of UEs and 'P' is the
        % number of RBs in the bandwidth. A matrix element at position (i,
        % j) corresponds to CQI value for UE with RNTI 'i' at RB 'j'
        CQIInfo = cell(1, 1);

        %CQIVisualizationGridHandles Handles to display UE CQIs on the RBs of the bandwidth
        CQIVisualizationGridHandles

        %CQIMapHandle Handle of the CQI heat map
        CQIMapHandle

        %CVCurrView Type of channel quality displayed in CQI
        % Visualization. Value 1 represents downlink and value 2 represents
        % uplink
        CVCurrView = 1

        %CVLowerUEIndex Index of the first UE to be displayed in CQI visualization
        CVLowerUEIndex = 0

        %CVUpperUEIndex Index of the last UE to be displayed in CQI visualization
        CVUpperUEIndex

        %CVLowerRBIndex Index of the first RB to be displayed in CQI visualization
        CVLowerRBIndex = 0

        %CVUpperRBIndex Index of the last RB to be displayed in CQI visualization
        CVUpperRBIndex

        %CVTxtHandle UI control handle to display the frame number in CQI visualization
        CVTxtHandle

        %UENames Names of the UEs
        UENames
    end

    properties(Hidden)
        %IsLegendRequired Flag to control the GUI elements on the grid
        IsLegendRequired = false;

        %HAxis RB grid visualization UI axis
        HAxis

        %AlertBoxTitle Title for the alert box
        AlertBoxTitle = "Grid Visualizer"
    end

    methods
        function obj = helperNRGridVisualizer(numFrameSim, gNB, UEs, varargin)
            %helperNRGridVisualizer Construct scheduling log visualization object
            %
            % OBJ = helperNRGridVisualizer(NUMFRAMESIM, GNB, UES) Create
            % grid visualization object.
            %
            % NUMFRAMESSIM is simulation time in terms of number of 10 ms frames.
            %
            % GNB is an object of type nrGNB.
            %
            % UEs is an array of node objects of type nrUE. They must be connected to
            % the same GNB.

            % Initialize the properties
            for idx = 1:2:numel(varargin)
                obj.(varargin{idx}) = varargin{idx+1};
            end

            % Validate number of frames in simulation
            obj.NumFrames = numFrameSim;

            if ~isempty(obj.IsLogReplay) && ~isempty(obj.SimulationLogs) && obj.IsLogReplay == 0
                cellChanged(obj, obj.SimulationLogs{1}.CellName);
            else
                updateContext(obj, gNB, UEs);
            end

            if ~obj.IsLegendRequired
                setupGUI(obj);
            end
        end

        function updateContext(obj, gNB, UEs)

            obj.NumUEs = numel(UEs);
            obj.UENames = [UEs.Name];
            obj.NumHARQ = gNB.NumHARQ;
            obj.NumSlotsFrame = (10 * gNB.SubcarrierSpacing*1e-3) / 15; % Number of slots in a 10 ms frame

            % Verify Duplex mode and update the properties
            if strcmpi(gNB.DuplexMode, "TDD")
                obj.DuplexMode = 1;
            end
            if obj.DuplexMode == obj.TDDDuplexMode % TDD
                obj.NumLogs = 1;
                obj.RVCurrView = 1; % Only one view for resource grid
            end

            % Determine the plots
            % Downlink & Uplink
            obj.PlotIds = [obj.DownlinkIdx obj.UplinkIdx];
            % Show the enabled visualization as current view
            if obj.LinkDirection ~= 2
                obj.PlotIds = obj.LinkDirection+1;
                obj.RVCurrView = obj.PlotIds;
                obj.CVCurrView = obj.PlotIds;
            end

            % Initialize number of RBs, CQI and metrics properties
            for idx = 1: numel(obj.PlotIds)
                logIdx = obj.PlotIds(idx);
                obj.NumRBs(logIdx) = gNB.NumResourceBlocks; % Number of RBs in DL/UL
            end
            obj.CQIInfo = zeros(obj.NumUEs, obj.NumRBs(logIdx)); % DL channel quality

            if obj.SchedulingType % Symbol based scheduling
                gridLength = obj.NumSym;
            else % Slot based scheduling
                gridLength = obj.NumSlotsFrame;
            end

            % Initialize the scheduling logs and resources grid related
            % properties
            for idx=1:min(obj.NumLogs,numel(obj.PlotIds))
                plotId = obj.PlotIds(idx);
                if obj.DuplexMode == obj.FDDDuplexMode
                    logIdx = plotId; % FDD
                else
                    logIdx = idx; % TDD
                end
                % Construct the log format
                obj.ResourceGrid{logIdx} = zeros(gridLength, obj.NumRBs(plotId));
                obj.ResourceGridReTxInfo{logIdx} = zeros(gridLength, obj.NumRBs(plotId));
                obj.ResourceGridHarqInfo{logIdx} = zeros(gridLength, obj.NumRBs(plotId));
                obj.ResourceGridInfo{logIdx} = strings(gridLength, obj.NumRBs(plotId));
            end
            obj.RGLowerRBIndex = 0;
            obj.RGLowerSlotIndex = 0;
            obj.CVLowerUEIndex = 0;
            obj.CVLowerRBIndex = 0;
        end

        function plotCQIRBGrids(obj, varargin)
            %plotCQIRBGrids Updates the CQI grid visualization
            %
            % plotCQIRBGrids(OBJ, SIMSLOTNUM) To update the CQI
            % grid and CQI visualization in live visualization
            %
            % SIMSLOTNUM - Cumulative slot number in the simulation

            % Update frame number in the figure (in live visualization)
            if isempty(obj.IsLogReplay)
                slotNum = varargin{1};
                obj.CurrFrame = floor(slotNum / obj.NumSlotsFrame)-1;
                obj.CurrSlot = mod(slotNum-1, obj.NumSlotsFrame);
            end
            updateCQIVisualization(obj);
            drawnow;
        end

        function plotRBGrids(obj, varargin)
            %plotRBGrids Updates the resource grid visualization
            %
            % plotRBGrids(OBJ) To update the resource
            % grid and CQI visualization in post-simulation visualization
            %
            % plotRBGrids(OBJ, SIMSLOTNUM) To update the resource
            % grid and CQI visualization in live visualization
            %
            % SIMSLOTNUM - Cumulative slot number in the simulation

            % Check if the figure handle is valid
            if isempty(obj.RGVisualizationFigHandle) || ~ishghandle(obj.RGVisualizationFigHandle)
                return;
            end

            % Update frame number in the figure (in live visualization)
            if isempty(obj.IsLogReplay)
                slotNum = varargin{1};
                obj.CurrFrame = floor((slotNum-1) / obj.NumSlotsFrame);
                if obj.CurrFrame < 0
                    obj.CurrFrame = 0;
                end
                obj.CurrSlot = mod(slotNum-1, obj.NumSlotsFrame);
            end

            if isempty(obj.CurrFrame)
                return;
            end
            if obj.DuplexMode == obj.TDDDuplexMode
                [obj.ResourceGrid, obj.ResourceGridReTxInfo, obj.ResourceGridHarqInfo, obj.SymSlotInfo] = obj.SchedulingLogger.getRBGridsInfo(obj.CurrFrame, obj.CurrSlot);
            else
                [obj.ResourceGrid, obj.ResourceGridReTxInfo, obj.ResourceGridHarqInfo] = obj.SchedulingLogger.getRBGridsInfo(obj.CurrFrame, obj.CurrSlot);
            end
            for idx = 1:min(obj.NumLogs, numel(obj.PlotIds))
                plotId = obj.PlotIds(idx);
                if obj.DuplexMode == obj.FDDDuplexMode
                    logIdx = obj.PlotIds(idx);
                else
                    logIdx = 1;
                end

                slIdx = size(obj.ResourceGrid{logIdx}, 1);
                for p = 1:slIdx
                    for q = 1 : obj.NumRBs(plotId)
                        if(obj.ResourceGrid{logIdx}(p, q) == 0)
                            % Clear the previously plotted text in the resource grid
                            obj.ResourceGridInfo{logIdx}(p, q)  = '';
                        else
                            % Create the text to be plotted in the resource
                            % grid
                            obj.ResourceGridInfo{logIdx}(p, q) = ...
                                obj.UENames(obj.ResourceGrid{logIdx}(p, q)) + "(" + obj.ResourceGridHarqInfo{logIdx}(p, q) + ")";
                        end
                    end
                end
            end
            updateResourceGridVisualization(obj);
        end

        function constructCQIGridVisualization(obj, varargin)
            %constructCQIGridVisualization Construct CQI grid visualization
            %
            % constructCQIGridVisualization(OBJ, Info) Construct CQI grid visualization
            %
            % Info - Info can be figure handle or a logical value. If it is a figure
            % handle, it is used for plotting. If it is logical value existing figure
            % is used.

            maxRBs = max(obj.NumRBs(obj.PlotIds));
            updateLegend = true;
            if nargin == 2
                if islogical(varargin{1})
                    updateLegend = varargin{1};
                    g = obj.CQIVisualizationFigHandle.Children;
                else
                    obj.CQIVisualizationFigHandle = varargin{1};
                    obj.CQIGridVisualization = true;
                end
            end
            compCounter = 2;
            if updateLegend
                g = uigridlayout(obj.CQIVisualizationFigHandle);
                g.RowHeight = {'fit','fit','fit','fit','fit','fit','fit','1x'};
                g.ColumnWidth = {'fit','fit','1x','1x'};

                numCells = numel(obj.SimulationLogs);
                if numCells > 1
                    [~, itemsData] = constructCellItemList(obj, numCells);
                    for idx=1:numCells
                        cellNames(idx) = obj.SimulationLogs{idx}.CellName;
                    end
                    lb1 = uilabel(g,'Text','Select Cell:');
                    lb1.Layout.Row = compCounter;
                    lb1.Layout.Column = 1;
                    dd = uidropdown(g,'Items',cellNames, 'ItemsData', itemsData, 'ValueChangedFcn', @(dd, event) cellChanged(obj, dd.Items{dd.Value}));
                    dd.Layout.Row = compCounter;
                    dd.Layout.Column = 2;
                end
                compCounter = compCounter + 1;

                if obj.LinkDirection == 2
                    lb1 = uilabel(g,'Text','Selected Link: ');
                    lb1.Layout.Row = compCounter;
                    lb1.Layout.Column = 1;

                    % Link direction
                    lb1Val = uilabel(g, 'Text', 'Downlink');
                    lb1Val.Layout.Row = compCounter;
                    lb1Val.Layout.Column = 2;
                else
                    compCounter = 2;
                end
            else
                compCounter = 3;
            end
            if obj.CVMaxRBsToDisplay <= maxRBs
                compCounter = compCounter + 1;
                obj.CVUpperRBIndex = obj.CVMaxRBsToDisplay;
                lb2 = uilabel(g,'Text','Select RB Range:');
                lb2.Layout.Row = compCounter;
                lb2.Layout.Column = 1;
                [items, itemsData] = constructRBItemList(obj, obj.NumRBs(obj.CVCurrView));
                dd2 = uidropdown(g,'Items',items, 'ItemsData', itemsData, 'ValueChangedFcn', @(dd, event) cbSelectedRBRange(obj, dd.Value));
                dd2.Layout.Row = compCounter;
                dd2.Layout.Column = 2;
            else
                obj.CVUpperRBIndex = maxRBs;
            end

            % Number of UEs to be displayed in the default view of CQI visualization
            if obj.NumUEs >= obj.CVMaxUEsToDisplay
                compCounter = compCounter + 1;
                obj.CVUpperUEIndex = obj.CVMaxUEsToDisplay;
                obj.CVUpperRBIndex = obj.CVMaxRBsToDisplay;
                lb2 = uilabel(g,'Text','Select UE Range:');
                lb2.Layout.Row = compCounter;
                lb2.Layout.Column = 1;
                [items, itemsData] = cvDropDownForUERange(obj);
                dd2 = uidropdown(g,'Items',items, 'ItemsData', itemsData, 'ValueChangedFcn', @(dd, event) cbSelectedUERange(obj, dd.Value));
                dd2.Layout.Row = compCounter;
                dd2.Layout.Column = 2;
            else
                obj.CVUpperUEIndex = obj.NumUEs;
            end

            % If post simulation log analysis enabled
            if isempty(obj.IsLogReplay) || obj.IsLogReplay
                % Create label for frame number
                compCounter = compCounter + 1;
                lb3 = uilabel(g, 'Text', 'Frame Number:');
                lb3.Layout.Row = compCounter;
                lb3.Layout.Column = 1;
                obj.CVTxtHandle = uilabel(g, 'Text', ' ');
                obj.CVTxtHandle.Layout.Row = compCounter;
                obj.CVTxtHandle.Layout.Column = 2;
            else
                if obj.IsLegendRequired
                    compCounter = compCounter + 1;
                    lb3 = uilabel(g, 'Text', 'Total Frames:');
                    lb3.Layout.Row = compCounter;
                    lb3.Layout.Column = 1;
                    lb3 = uilabel(g, 'Text', num2str(obj.NumFrames));
                    lb3.Layout.Row = compCounter;
                    lb3.Layout.Column = 2;
                end
                compCounter = compCounter + 1;
                lb4  = uilabel(g, 'Text','Frame number:');
                lb4.Layout.Row = compCounter;
                lb4.Layout.Column = 1;
                if obj.IsLegendRequired
                    obj.CVTxtHandle = uieditfield(g, 'numeric', 'Value', 0, 'ValueChangedFcn', @(dd, event) showFrame(obj, dd.Value),'Limits', [0 obj.NumFrames-1]);
                else
                    obj.CVTxtHandle = uilabel(g, 'Text', ' ');
                end
                obj.CVTxtHandle.Layout.Row = compCounter;
                obj.CVTxtHandle.Layout.Column = 2;
            end

            % Construct the CQI map
            title = '';
            if ~obj.IsLegendRequired
                title = ['Channel  Quality Visualization for Cell ID - '  num2str(obj.CellOfInterest)];
            end
            if obj.CVMaxRBsToDisplay <= maxRBs
                obj.CVUpperRBIndex = obj.CVMaxRBsToDisplay;
            else
                obj.CVUpperRBIndex = maxRBs;
            end

            % Number of UEs to be displayed in the default view of CQI visualization
            if obj.CVMaxUEsToDisplay <= obj.NumUEs
                obj.CVUpperUEIndex = obj.CVMaxUEsToDisplay;
            else
                obj.CVUpperUEIndex = obj.NumUEs;
            end
            numRBsToDisplay = obj.CVUpperRBIndex - obj.CVLowerRBIndex;
            numUEsToDisplay = obj.CVUpperUEIndex - obj.CVLowerUEIndex;
            obj.CQIMapHandle = heatmap(g, zeros(numRBsToDisplay, numUEsToDisplay), ...
                'CellLabelColor', 'none', 'XLabel', 'UEs', 'YLabel', ...
                'Resource Blocks', 'ColorLimits', [0 15], 'Title', title, 'Colormap', parula(16),'GridVisible',true);

            % Set CQI-visualization axis label
            updateCQIMapProperties(obj);

            % Set the layout
            obj.CQIMapHandle.Layout.Row = [1 8];
            obj.CQIMapHandle.Layout.Column = [3 4];
        end

        function constructResourceGridVisualization(obj, varargin)
            %constructResourceGridVisualization Construct resource grid visualization
            %
            % constructResourceGridVisualization(OBJ, Info) Construct resource grid visualization
            %
            % Info - Info can be figure handle or a logical value. If it is a figure
            % handle, it is used for plotting. If it is logical value existing figure
            % is used.

            import matlab.graphics.internal.themes.specifyThemePropertyMappings
            maxRBs = max(obj.NumRBs(obj.PlotIds));
            updateLegend = true;
            if nargin == 2
                if islogical(varargin{1})
                    updateLegend = varargin{1};
                    g = obj.RGVisualizationFigHandle.Children;
                else
                    obj.RGVisualizationFigHandle = varargin{1};
                    obj.ResourceGridVisualization = true;
                end
            end

            if updateLegend % Update the cell specific legend information
                g = uigridlayout(obj.RGVisualizationFigHandle);
                obj.HAxis = uiaxes(g, 'Clipping','on');
                if ~obj.IsLegendRequired
                    obj.HAxis.Title = text("String",['Resource Grid Allocation for Cell ID - '  num2str(obj.CellOfInterest)]);
                end
                compCounter = 2;
                g.ColumnWidth = {100,100,'1x'};
                g.RowHeight = {'fit','fit','fit','fit','fit','fit','fit','fit','fit','fit','fit','fit','fit','fit','1x'};

                lb1 = uilabel(g,'Text','UE(n) : Transmission');
                lb1.Layout.Row = compCounter;
                compCounter = compCounter + 1;
                lb1.Layout.Column = [1 2];
                lb1 = uilabel(g,'Text','UE(n) : Retransmission');
                specifyThemePropertyMappings(lb1,FontColor="--mw-graphics-colorOrder-1-quaternary")
                lb1.Layout.Row = compCounter;
                compCounter = compCounter + 1;
                lb1.Layout.Column = [1 2];
                lb1 = uilabel(g,'Text','UE : UE Name');
                lb1.Layout.Row = compCounter;
                compCounter = compCounter + 1;
                lb1.Layout.Column = [1 2];
                lb1 = uilabel(g,'Text', 'n : HARQ Process ID');
                lb1.Layout.Row = compCounter;
                compCounter = compCounter + 1;
                lb1.Layout.Column = [1 2];

                numCells = numel(obj.SimulationLogs);
                if numCells > 1
                    compCounter = compCounter + 1;
                    [~, itemsData] = constructCellItemList(obj, numCells);
                    for idx=1:numCells
                        cellNames(idx) = obj.SimulationLogs{idx}.CellName;
                    end
                    lb1 = uilabel(g,'Text','Select Cell:');
                    lb1.Layout.Row = compCounter;
                    lb1.Layout.Column = 1;
                    dd = uidropdown(g,'Items',cellNames, 'ItemsData', itemsData, 'ValueChangedFcn', @(dd, event) cellChanged(obj, dd.Items{dd.Value}));
                    dd.Layout.Row = compCounter;
                    dd.Layout.Column = 2;
                end

                % Create drop-down for link type
                if min(obj.NumLogs, numel(obj.PlotIds))== 2
                    compCounter = compCounter + 1;
                    lb1 = uilabel(g,'Text','Select Link:');
                    lb1.Layout.Row = compCounter;
                    lb1.Layout.Column = 1;
                    dd1 = uidropdown(g,'Items',{'Downlink','Uplink'}, 'ItemsData', obj.PlotIds, 'ValueChangedFcn', @(dd, event) rbSelectedLinkType(obj, dd.Value, obj.HAxis));
                    dd1.Layout.Row = compCounter;
                    dd1.Layout.Column = 2;
                end
            else
                compCounter = 8;
            end

            % Construct drop down menu for RB range
            if obj.RGMaxRBsToDisplay <= maxRBs
                compCounter = compCounter + 1;
                lb2 = uilabel(g,'Text','Select RB Range:');
                lb2.Layout.Row = compCounter;
                lb2.Layout.Column = 1;
                [items, itemsData] = constructRBItemList(obj, obj.NumRBs(obj.RVCurrView));
                dd2 = uidropdown(g,'Items',items, 'ItemsData', itemsData, 'ValueChangedFcn', @(dd, event) rgSelectedRBRange(obj, dd.Value, obj.HAxis));
                dd2.Layout.Row = compCounter;
                dd2.Layout.Column = 2;
            end

            % Construct the drop-down item list
            for idx = 1:min(obj.NumLogs, numel(obj.PlotIds))
                if obj.DuplexMode == obj.FDDDuplexMode
                    plotId = obj.PlotIds(idx);
                else
                    plotId = idx;
                end
                % Construct the drop down based on number of RBs
                if obj.RGMaxRBsToDisplay < obj.NumRBs(plotId)
                    [obj.RBItemsList{plotId}, ~] = constructRBItemList(obj, obj.NumRBs(plotId));
                end
            end


            % If post simulation log analysis enabled
            if isempty(obj.IsLogReplay) || obj.IsLogReplay
                compCounter = compCounter + 1;
                % Create label for frame number
                lb3 = uilabel(g, 'Text', 'Frame Number:');
                lb3.Layout.Row = compCounter;
                lb3.Layout.Column = 1;
                obj.RGTxtHandle  = uilabel(g, 'Text', '');
                obj.RGTxtHandle.Layout.Row = compCounter;
                obj.RGTxtHandle.Layout.Column = 2;
                if obj.SchedulingType % Symbol based scheduling
                    compCounter = compCounter + 1;
                    % Create label for slot number
                    lb3 = uilabel(g, 'Text', 'Slot Number:');
                    lb3.Layout.Row = compCounter;
                    lb3.Layout.Column = 1;
                    obj.RGSlotTxtHandle = uilabel(g, 'Text', '');
                    obj.RGSlotTxtHandle.Layout.Row = compCounter;
                    obj.RGSlotTxtHandle.Layout.Column = 2;
                end
            else
                compCounter = compCounter + 1;
                lb3 = uilabel(g, 'Text', 'Total Frames:');
                lb3.Layout.Row = compCounter;
                lb3.Layout.Column = 1;
                lb3 = uilabel(g, 'Text', num2str(obj.NumFrames));
                lb3.Layout.Row = compCounter;
                lb3.Layout.Column = 2;
                compCounter = compCounter + 1;
                lb4  = uilabel(g, 'Text','Frame number: ');
                lb4.Layout.Row = compCounter;
                lb4.Layout.Column = 1;
                obj.RGTxtHandle = uieditfield(g, 'numeric', 'Value' , 0, 'ValueChangedFcn', @(dd, event) showFrame(obj, dd.Value),'Limits', [0 obj.NumFrames-1]);
                obj.RGTxtHandle.Layout.Row = compCounter;
                obj.RGTxtHandle.Layout.Column = 2;
                if obj.SchedulingType % Symbol based scheduling
                    compCounter = compCounter + 1;
                    lb4  = uilabel(g, 'Text','Slot number:');
                    lb4.Layout.Row = compCounter;
                    lb4.Layout.Column = 1;
                    obj.RGSlotTxtHandle = uieditfield(g, 'numeric', 'Value' , 0, 'ValueChangedFcn', @(dd, event) showSlot(obj, dd.Value),'Limits', [0 obj.NumSlotsFrame-1]);
                    obj.RGSlotTxtHandle.Layout.Row = compCounter;
                    obj.RGSlotTxtHandle.Layout.Column = 2;
                    obj.CurrFrame  = 0;
                    obj.CurrSlot = 0;
                end
            end

            if obj.SchedulingType == obj.SlotBased && obj.RGMaxSlotsToDisplay < obj.NumSlotsFrame
                compCounter = compCounter + 1;
                % Create drop-down for Slot range
                lb2 = uilabel(g,'Text','Slot Range:');
                lb2.Layout.Row = compCounter;
                lb2.Layout.Column = 1;
                [items, itemsData] = rgDropDownForSlotRange(obj);
                dd2 = uidropdown(g,'Items',items, 'ItemsData', itemsData, 'ValueChangedFcn', @(dd, event) rgSelectedSlotRange(obj, dd.Value, obj.HAxis));
                dd2.Layout.Row = compCounter;
                dd2.Layout.Column = 2;
            end

            % Number of RBs to be displayed in the default view of resource grid visualization
            if obj.RGMaxRBsToDisplay <= maxRBs
                obj.RGUpperRBIndex = obj.RGMaxRBsToDisplay;
            else
                obj.RGUpperRBIndex = maxRBs;
            end
            % Number of slots to be displayed in the default view of resource grid visualization
            if obj.NumSlotsFrame >= obj.RGMaxSlotsToDisplay
                obj.RGUpperSlotIndex = obj.RGMaxSlotsToDisplay;
            else
                obj.RGUpperSlotIndex = obj.NumSlotsFrame;
            end

            % Set axis properties
            obj.HAxis.Layout.Column = 3;
            obj.HAxis.Layout.Row = [2 15];
            drawnow;
            obj.ResourceGridTextHandles  = gobjects(obj.NumSlotsFrame, maxRBs);

            if obj.SchedulingType
                % Initialize the symbol pattern in a slot
                for sidx =1:obj.NumSym
                    obj.SymSlotInfo{sidx} = "Symbol-" + (sidx-1);
                end
            else
                % Initialize the slot pattern in a frame
                for sidx =1:obj.NumSlotsFrame
                    obj.SymSlotInfo{sidx} = "Slot-" + (sidx-1);
                end
            end

            % Set resource-grid visualization axis label
            replotResourceGrid(obj, obj.HAxis, 'XAxis');
            if obj.SchedulingType
                xlabel(obj.HAxis, 'Symbols in Slot');
            else
                xlabel(obj.HAxis, 'Slots in 10 ms Frame');
            end
            replotResourceGrid(obj, obj.HAxis, 'YAxis');
            ylabel(obj.HAxis, 'Resource Blocks');
            obj.HAxis.TickDir = 'out';
            drawnow;
        end

        function updateCQIVisualization(obj)
            %updateCQIVisualization Update the CQI map

            if isempty(obj.CurrFrame)
                return;
            end

            % Check if the figure handle is valid
            if ~obj.IsLegendRequired && (isempty(obj.CQIVisualizationFigHandle) || ~ishghandle(obj.CQIVisualizationFigHandle))
                return;
            end

            if obj.SchedulingType == obj.SlotBased
                obj.CurrSlot = obj.NumSlotsFrame - 1;
            end

            if ~obj.IsLegendRequired
                obj.CVTxtHandle.Text = num2str(obj.CurrFrame);
            end

            % Get the CQI information
            obj.CQIInfo = obj.SchedulingLogger.getCQIRBGridsInfo(obj.CurrFrame, obj.CurrSlot);
            % Make the CQI Map grid structure similar to RBG map
            obj.CQIMapHandle.ColorData = flipud(obj.CQIInfo(obj.CVLowerUEIndex+1:obj.CVUpperUEIndex, obj.CVLowerRBIndex+1:obj.CVUpperRBIndex)');
            drawnow;
        end

        function updateResourceGridVisualization(obj)
            %updateResourceGridVisualization Update the resource grid visualization

            import matlab.graphics.internal.themes.specifyThemePropertyMappings
            if isempty(obj.IsLogReplay) || obj.IsLogReplay == 1
                if isempty(obj.CurrFrame)
                    obj.RGTxtHandle.Text = "";
                else
                    obj.RGTxtHandle.Text = "" + obj.CurrFrame; % Update the frame number
                end
            end
            if obj.SchedulingType % For symbol based scheduling
                lowLogIdx = 0;
                uppLogIdx = obj.NumSym;
                % Update the axis
                obj.RGVisualizationFigHandle.CurrentAxes.XTickLabel = obj.SymSlotInfo;
                if isempty(obj.IsLogReplay) || obj.IsLogReplay == 1
                    if isempty(obj.CurrSlot)
                        obj.RGSlotTxtHandle.Text = "";
                    else
                        obj.RGSlotTxtHandle.Text = "" + obj.CurrSlot; % Update the slot number
                    end
                end
            else % For slot based scheduling
                lowLogIdx = obj.RGLowerSlotIndex;
                uppLogIdx = obj.RGUpperSlotIndex;
                % Update the axis
                obj.RGVisualizationFigHandle.CurrentAxes.XTickLabel = obj.SymSlotInfo(obj.RGLowerSlotIndex+1 : obj.RGUpperSlotIndex);
            end
            for n = lowLogIdx+1 : uppLogIdx
                for p = obj.RGLowerRBIndex + 1 : obj.RGUpperRBIndex
                    obj.ResourceGridTextHandles(n, p).String = obj.ResourceGridInfo{obj.RVCurrView}(n, p);
                    if(obj.ResourceGridReTxInfo{obj.RVCurrView}(n, p) == 2) % Re-Tx
                        specifyThemePropertyMappings(obj.ResourceGridTextHandles(n, p),Color="--mw-graphics-colorOrder-1-quaternary");
                    else
                        specifyThemePropertyMappings(obj.ResourceGridTextHandles(n, p),Color="--mw-graphics-colorNeutral-line-tertiary");
                    end
                end
            end
            drawnow;
        end

        function plotPostSimRBGrids(obj, simSlotNum)
            %plotPostSimRBGrids Post simulation log visualization
            %
            % plotPostSimRBGrids(OBJ, SIMSLOTNUM) To update the resource
            % grid and CQI visualization based on the post simulation logs.
            %
            % SIMSLOTNUM - Cumulative slot number in the simulation

            % Update slot number
            if obj.SchedulingType % Symbol based scheduling
                obj.CurrSlot = mod(simSlotNum-1, obj.NumSlotsFrame);
                if obj.CurrSlot == 0
                    obj.CurrFrame = floor(simSlotNum/obj.NumSlotsFrame);
                end
            else % Slot based scheduling
                obj.CurrSlot = obj.NumSlotsFrame - 1;
                obj.CurrFrame = floor(simSlotNum/obj.NumSlotsFrame) - 1;
            end

            % Update grid information at slot boundary (for symbol based
            % scheduling) and frame boundary (for slot based scheduling)
            % Update resource grid visualization
            plotRBGrids(obj);
            % Update CQI visualization
            plotCQIRBGrids(obj);
        end

        function showFrame(obj, frameNumber)
            %showFrame Handle the event when user enters a
            % number to visualize a particular frame number in the
            % simulation

            % Update the resource grid and CQI grid visualization
            try
                validateattributes(frameNumber, {'numeric'}, {'real', 'integer', 'scalar'});
            catch
                if ~obj.IsLegendRequired || obj.ResourceGridVisualization
                    figure = obj.RGVisualizationFigHandle;
                    obj.RGTxtHandle.Value = obj.CurrFrame;
                else
                    figure = obj.CQIVisualizationFigHandle;
                    obj.CVTxtHandle.Value = obj.CurrFrame;
                end
                uialert(figure,"'Frame number' value must be a positive integer.",obj.AlertBoxTitle,'Interpreter','html');
                return;
            end

            obj.CurrFrame = frameNumber;
            if obj.CQIGridVisualization
                updateCQIVisualization(obj);
            end
            if obj.ResourceGridVisualization
                plotRBGrids(obj);
            end
        end

        function showSlot(obj, slotNumber)
            %showFrame Handle the event when user enters a
            % number to visualize a particular slot number in the
            % simulation

            try
                validateattributes(slotNumber, {'numeric'}, {'real', 'integer', 'scalar'});
            catch
                obj.RGSlotTxtHandle.Value = obj.CurrSlot;
                uialert(obj.RGVisualizationFigHandle,"'Slot number' value must be a positive integer.",obj.AlertBoxTitle,'Interpreter','html');
                return;
            end
            obj.CurrSlot = slotNumber;
            % Update the resource grid and CQI grid visualization
            if obj.ResourceGridVisualization
                plotRBGrids(obj);
            end
        end
    end

    methods(Access = public)
        function [itemList, itemData] = constructCellItemList(obj, numCells)
            %constructCellItemList Create the items for the drop-down component

            % Create the items for the drop-down component
            itemList = cell(numCells, 1);
            itemData = zeros(numCells, 1);
            for i = 1 : numCells
                itemData(i) = i;
                itemList{i} = ['Cell - ', num2str(mod(i, obj.MaxCells))];
            end
        end

        function [itemList, itemData] = constructRBItemList(obj, numRBs)
            %constructRBItemList Create the items for the drop-down component

            % Create the items for the drop-down component
            numItems = floor(numRBs / obj.RGMaxRBsToDisplay);
            itemList = cell(numItems, 1);
            itemData = zeros(ceil(numRBs / obj.RGMaxRBsToDisplay), 1);
            for i = 1 : numItems
                itemData(i) = (i - 1) * obj.RGMaxRBsToDisplay;
                itemList{i} = ['RB ', num2str(itemData(i)) '-' num2str(itemData(i) + obj.RGMaxRBsToDisplay - 1)];
            end
            if (mod(numRBs,obj.RGMaxRBsToDisplay) > 0)
                itemData(i+1) = i * obj.RGMaxRBsToDisplay;
                itemList{i+1} = ['RB ', num2str(itemData(i+1)) '-' num2str(numRBs - 1)];
            end
        end

        function [itemList, itemData] = rgDropDownForSlotRange(obj)
            %rgDropDownForSlotRange Construct drop-down component for selecting slot range

            % Create the items for the drop-down component
            numItems = floor(obj.NumSlotsFrame / obj.RGMaxSlotsToDisplay);
            itemData = zeros(ceil(obj.NumSlotsFrame / obj.RGMaxSlotsToDisplay), 1);
            itemList = cell(numItems, 1);
            for i = 1 : numItems
                itemData(i) = (i-1) * obj.RGMaxSlotsToDisplay ;
                itemList{i} = ['Slot ', num2str(itemData(i)) '-' num2str(itemData(i) + obj.RGMaxSlotsToDisplay - 1)];
            end
            if (mod(obj.NumSlotsFrame, obj.RGMaxSlotsToDisplay) > 0)
                itemData(i+1) = i * obj.RGMaxSlotsToDisplay + 1;
                itemList{i+1} = ['Slot ', num2str(itemData(i+1) - 1) '-' num2str(obj.NumSlotsFrame - 1)];
            end
        end

        function [itemList, itemData] = cvDropDownForUERange(obj)
            %cvDropDownForUERange Construct drop-down component for selecting UEs

            % Create the items for the drop-down component
            numItems = floor(obj.NumUEs / obj.CVMaxUEsToDisplay);
            itemData = zeros(ceil(obj.NumUEs / obj.CVMaxUEsToDisplay), 1);
            itemList = cell(numItems, 1);
            for i = 1 : numItems
                itemData(i) = (i - 1) * obj.CVMaxUEsToDisplay;
                itemList{i} = ['UE ', num2str(itemData(i) + 1) '-' num2str(itemData(i) + obj.CVMaxUEsToDisplay)];
            end
            if (mod(obj.NumUEs,obj.CVMaxUEsToDisplay) > 0)
                itemData(i+1) = i * obj.CVMaxUEsToDisplay;
                itemList{i+1} = ['UE ', num2str(itemData(i+1)+1) '-' num2str(itemData(i+1) + mod(obj.NumUEs, obj.CVMaxUEsToDisplay))];
            end
        end

        function cellChanged(obj, cellName)
            %cellChanged Handle the event when user selects a cell

            for idx=1:numel(obj.SimulationLogs)
                if obj.SimulationLogs{idx}.CellName == cellName
                    break;
                end
            end
            logInfo = obj.SimulationLogs{idx};
            updateContext(obj, logInfo.GNB, logInfo.UEs)
            logObj = helperNRSchedulingLogger(logInfo.NumFramesSim, logInfo.GNB, logInfo.UEs, IsLogReplay=0);
            if strcmpi(logInfo.GNB.DuplexMode,"TDD") % TDD
                logObj.SchedulingLog{1} = logInfo.TimeStepLogs(2:end,:);
            else % FDD
                logObj.SchedulingLog{1} = logInfo.DLTimeStepLogs(2:end,:);
                logObj.SchedulingLog{2} = logInfo.ULTimeStepLogs(2:end,:);
            end
            obj.SchedulingLogger = logObj;
            if obj.ResourceGridVisualization
                % Delete the old components and add new componenets related
                % to the selected cell configuration
                if obj.DuplexMode
                    idx = 8;
                else
                    idx=10;
                end
                delete(obj.RGVisualizationFigHandle.Children.Children(idx:end));
                constructResourceGridVisualization(obj, false);
            end
            if obj.CQIGridVisualization
                delete(obj.CQIVisualizationFigHandle.Children.Children(6:end));
                constructCQIGridVisualization(obj, false);
            end
            showFrame(obj, 0);
        end

        function rgSelectedRBRange(obj, lowerRBIndex, hAx)
            %rgSelectedRBRange Handle the event when user selects RB range in resource grid visualization

            obj.RGLowerRBIndex = lowerRBIndex;
            obj.RGUpperRBIndex = obj.RGLowerRBIndex + obj.RGMaxRBsToDisplay;
            if obj.RGUpperRBIndex > obj.NumRBs(obj.RVCurrView)
                obj.RGUpperRBIndex = obj.NumRBs(obj.RVCurrView);
            end
            % Update the Y-Axis of the resource grid visualization with
            % selected RB range
            replotResourceGrid(obj, hAx, 'YAxis');
        end

        function rgSelectedSlotRange(obj, lowerSlotIndex, hAx)
            %rgSelectedSlotRange Handle the event when user selects slot range in resource grid visualization

            obj.RGLowerSlotIndex = lowerSlotIndex;
            obj.RGUpperSlotIndex = obj.RGLowerSlotIndex + obj.RGMaxSlotsToDisplay;
            if obj.RGUpperSlotIndex > obj.NumSlotsFrame
                obj.RGUpperSlotIndex = obj.NumSlotsFrame;
            end
            % Update the X-Axis of the resource grid visualization with
            % selected slot range
            replotResourceGrid(obj, hAx, 'XAxis');
        end

        function rbSelectedLinkType(obj, plotIdx, hAx)
            %rbSelectedLinkType Handle the event when user selects link type in resource grid visualization

            % Update the resource grid visualization with selected link type
            if numel(obj.PlotIds) == 2
                obj.RVCurrView = plotIdx;
            end
            replotResourceGrid(obj, hAx, 'YAxis');
            drawnow;
        end

        function replotResourceGrid(obj, hAx, coordinate)
            %replotResourceGrid Update the resource grid along X-axis or Y-axis w.r.t to the given input parameters.

            cla(hAx);
            numRBsToDisplay = obj.RGUpperRBIndex - obj.RGLowerRBIndex;
            if obj.SchedulingType % For symbol based scheduling
                lowLogIdx = 0;
                numUnitsToDisplay = obj.NumSym; % Display information of 14 symbols in a slot
            else % For slot based scheduling
                lowLogIdx = obj.RGLowerSlotIndex;
                numUnitsToDisplay = obj.RGUpperSlotIndex - obj.RGLowerSlotIndex;
            end
            [X1, Y1] = meshgrid(0:numUnitsToDisplay, 0 : numRBsToDisplay);
            [X2, Y2] = meshgrid(0:numRBsToDisplay, 0 : numUnitsToDisplay);
            x = linspace(1, numUnitsToDisplay, numUnitsToDisplay);
            y = linspace(1, numRBsToDisplay, numRBsToDisplay);
            for n=1:numUnitsToDisplay
                i = lowLogIdx + n;
                for p = 1 : numRBsToDisplay
                    j = obj.RGLowerRBIndex + p;
                    obj.ResourceGridTextHandles(i, j) = text(hAx, x(n) - .5, y(p) - .5, ' ', 'HorizontalAlignment', 'center', 'Clipping', 'on');
                end
            end
            hold(hAx, 'on');
            import matlab.graphics.internal.themes.specifyThemePropertyMappings
            h1 = plot(hAx, X1, Y1, 'LineWidth', 0.1);
            for i = 1:numel(h1)
                specifyThemePropertyMappings(h1(i),Color="--mw-color-readOnly")
            end
            h2 = plot(hAx, Y2, X2, 'LineWidth', 0.1);
            for i = 1:numel(h2)
                specifyThemePropertyMappings(h2(i),Color="--mw-color-readOnly")
            end
            if strcmpi('XAxis', coordinate) == 1
                % Updates X-Axis
                xticks(hAx, (1 : numUnitsToDisplay) - 0.5);
                if obj.SchedulingType % Symbol based scheduling
                    xticklabels(hAx, obj.SymSlotInfo);
                else
                    xticklabels(hAx, obj.SymSlotInfo(obj.RGLowerSlotIndex+1 : obj.RGUpperSlotIndex));
                end
            else
                % Updates Y-Axis
                yticks(hAx, (1 : numRBsToDisplay) - 0.5);
                yTicksLabel = cell(1, 0);
                for i = 1 : numRBsToDisplay
                    yTicksLabel{i} = "RB-" + (obj.RGLowerRBIndex+i-1);
                end
                yticklabels(hAx, yTicksLabel);
            end

            % Update the resource grid visualization
            updateResourceGridVisualization(obj);
        end

        function cbSelectedRBRange(obj, lowerRBIndex)
            %cbSelectedRBRange Handle the event when user selects RB range in CQI grid visualization

            obj.CVLowerRBIndex = lowerRBIndex;
            obj.CVUpperRBIndex = obj.CVLowerRBIndex + obj.CVMaxRBsToDisplay;
            if obj.CVUpperRBIndex > obj.NumRBs(obj.CVCurrView)
                obj.CVUpperRBIndex = obj.NumRBs(obj.CVCurrView);
            end
            % Update the Y-Axis limits of the CQI grid visualization with
            % selected RB range
            updateCQIMapProperties(obj);
        end

        function cbSelectedUERange(obj, lowerUEIndex)
            %cbSelectedUERange Handle the event when user selects UE range in CQI grid visualization

            obj.CVLowerUEIndex = lowerUEIndex;
            obj.CVUpperUEIndex = obj.CVLowerUEIndex + obj.CVMaxUEsToDisplay;
            if obj.CVUpperUEIndex  > obj.NumUEs
                obj.CVUpperUEIndex = obj.NumUEs;
            end
            % Update the X-Axis limits of the CQI grid visualization with
            % selected UE range
            updateCQIMapProperties(obj)
        end

        function cbSelectedLinkType(obj, plotIdx)
            %cbSelectedLinkType Handle the event when user selects link type in CQI grid visualization

            % Update the CQI grid visualization with selected link type
            if numel(obj.PlotIds) == 2
                obj.CVCurrView = plotIdx;
            end
            % Update the Y-Axis limits of the CQI grid visualization with
            % selected RB range
            updateCQIMapProperties(obj);
        end

        function updateCQIMapProperties(obj)
            %updateCQIMapProperties Update the CQI grid along X-axis or Y-axis w.r.t to the given input parameters

            numRBsToDisplay = obj.CVUpperRBIndex - obj.CVLowerRBIndex;
            numUEsToDisplay = obj.CVUpperUEIndex - obj.CVLowerUEIndex;
            obj.CQIMapHandle.ColorData = zeros(numRBsToDisplay, numUEsToDisplay);
            % Update X-Axis
            xTicksLabel = cell(numUEsToDisplay, 0);
            for i = 1:numUEsToDisplay
                xTicksLabel{i} = obj.UENames(obj.CVLowerUEIndex + i );
            end
            obj.CQIMapHandle.XDisplayLabels = xTicksLabel;

            % Update Y-Axis
            yTicksLabel = cell(numRBsToDisplay, 0);
            for i = 1 : numRBsToDisplay
                yTicksLabel{i} = "RB-" + (obj.CVLowerRBIndex+i-1);
            end
            obj.CQIMapHandle.YDisplayLabels = flip(yTicksLabel);
            updateCQIVisualization(obj);
        end

        function setupGUI(obj)
            %setupGUI Create the visualization for cell of interest

            % Using the screen width and height, calculate the figure width
            % and height
            resolution = get(0, 'ScreenSize');
            screenWidth = resolution(3);
            screenHeight = resolution(4);
            figureWidth = screenWidth * 0.90;
            figureHeight = screenHeight * 0.85;

            if obj.CQIGridVisualization % Create CQI visualization
                obj.CQIVisualizationFigHandle = uifigure('Name', 'Channel Quality Visualization', 'Position', [screenWidth * 0.05 screenHeight * 0.05 figureWidth figureHeight], 'HandleVisibility', 'on');
                % Use desktop theme to support dark theme mode
                matlab.graphics.internal.themes.figureUseDesktopTheme(obj.CQIVisualizationFigHandle);
                constructCQIGridVisualization(obj);
                if ~isempty(obj.SchedulingLogger)
                    addDepEvent(obj.SchedulingLogger, @obj.plotCQIRBGrids, obj.NumSlotsFrame); % Invoke for every frame
                end
            end

            if obj.ResourceGridVisualization % Create resource grid visualization
                obj.RGVisualizationFigHandle = uifigure('Name', 'Resource Grid Allocation', 'Position', [screenWidth * 0.05 screenHeight * 0.05 figureWidth figureHeight], 'HandleVisibility', 'on');
                % Use desktop theme to support dark theme mode
                matlab.graphics.internal.themes.figureUseDesktopTheme(obj.RGVisualizationFigHandle);
                constructResourceGridVisualization(obj);
                if ~isempty(obj.SchedulingLogger)
                    if obj.SchedulingType == obj.SymbolBased
                        addDepEvent(obj.SchedulingLogger, @obj.plotRBGrids, 1); % Invoke for every slot
                    else
                        addDepEvent(obj.SchedulingLogger, @obj.plotRBGrids, obj.NumSlotsFrame); % Invoke for every frame
                    end
                end
            end
        end
    end
end