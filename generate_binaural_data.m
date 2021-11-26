%Creating binaural audio versions
%---------------------------------------------------------------------------------------------
%Needed variables: 1. Path to TwoEars-master folder (Download TwoEars from http://twoears.eu/)
%                  2. Input path
%                  3. Output path
%---------------------------------------------------------------------------------------------
clear
folder_path = pwd;
output_path = 'your_output_path';
D = dir(folder_path);
for k = 3:length(D)
    currD = D(k).name;
    wavFiles = dir(fullfile(folder_path, currD, '*.wav'));
    for k=1:length(wavFiles)
        wavFile = wavFiles(k).name;
        wavPath = fullfile(folder_path, currD, wavFile);
        %start creating binarual versions
        for rad = 0:pi/4:(7*pi)/4
            addpath('your_TwoEars-master_path')
            startTwoEars
            sim = simulator.SimulatorConvexRoom();
            set(sim, ...
              'SampleRate', 44100, ...
              'HRIRDataset', simulator.DirectionalIR( ...
                'impulse_responses/qu_kemar_anechoic/QU_KEMAR_anechoic_1m.sofa'), ...
                'Sources', {simulator.source.Point()}, ...
                'Sinks', simulator.AudioSink(2) ...
              );
            % set parameters of audio sources
            set(sim.Sources{1}, ...
              'AudioBuffer', simulator.buffer.FIFO(1), ...
              'Position', [cos(rad); sin(rad); 0], ...
              'Name', currD ...
              );
            % set parameters of head
            set(sim.Sinks, ...
              'Position' , [0; 0; 0], ...
              'UnitX', [1; 0; 0], ...
              'Name', 'Head' ...
              );
            % set audio input of buffers
            set(sim.Sources{1}.AudioBuffer, ...
              'File', wavPath);
          
            sim.set('Init', true);
          
            while ~sim.isFinished()
              sim.set('Refresh',true);  % refresh all objects
              sim.set('Process',true);
            end
                    
          % Saves the file in the output directory
            data = sim.Sinks.getData();
            sim.Sinks.saveFile(fullfile(output_path, currD, strcat(wavFile(1:end-4), '_', num2str(rad2deg(rad)), '.wav')), sim.SampleRate);
            sim.set('ShutDown',true);

        end
        
        
    end

    
end
