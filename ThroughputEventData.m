classdef ThroughputEventData < event.EventData
    properties
        Throughput % Stores the throughput value

        Object

        TimeResource

        FrequencyResource

        SchedulingInfo
    end

    methods
        function obj = ThroughputEventData(throughput, object, timeResource, frequencyResource, schedulingInfo)
            obj.Throughput = throughput;
            obj.TimeResource = timeResource;
            obj.Object = object;
            obj.FrequencyResource = frequencyResource;
            obj.SchedulingInfo = schedulingInfo;
        end
    end
end