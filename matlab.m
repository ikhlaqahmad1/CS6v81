%% -----------------------------
% MATLAB Real-Time YOLO Coordinate Reader
% Reads coordinates from Python YOLO and visualizes them
%% -----------------------------

% Serial port settings (Python must send via serial)
pythonPort = "/dev/tty.usbmodem14101"; % Replace with your port
baudRate = 9600;
s = serialport(pythonPort, baudRate);
configureTerminator(s, "LF");
flush(s);

disp('🔵 Listening for coordinates from Python YOLO...');

% Camera settings (match your webcam resolution)
imgWidth = 640;
imgHeight = 480;

% Create a figure for visualization
figure;
hPlot = plot(nan, nan, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
xlim([0 imgWidth]);
ylim([0 imgHeight]);
xlabel('X'); ylabel('Y');
title('Object Tracking from Python YOLO');
set(gca, 'YDir','reverse'); % Reverse Y-axis to match image coordinates
grid on;

while true
    try
        % Read one line from Python
        data = readline(s);         % e.g., "320,240"
        coords = strsplit(strtrim(data), ',');

        if numel(coords) == 2
            cx = str2double(coords{1});
            cy = str2double(coords{2});

            fprintf('Received coordinates: X=%d, Y=%d\n', cx, cy);

            % Update plot
            set(hPlot, 'XData', cx, 'YData', cy);
            drawnow;
        end
    catch ME
        fprintf('⚠️ Error: %s\n', ME.message);
    end
end
