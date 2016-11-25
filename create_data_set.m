% dir_name = '/home/adutta/Workspace/SignatureVerification/Datasets/GPDS300/';
% subdir_names = dir(dir_name);
% 
% fp = fopen( 'GPDS300.txt','w');
% 
% for i = 1:length(subdir_names)
%     if( ~strcmp(subdir_names(i).name, '.') && ~strcmp(subdir_names(i).name, '.') && subdir_names(i).isdir )
%         file_names = dir( fullfile( dir_name, subdir_names(i).name, '/*.bmp' ) );
%         for j = 1:length(file_names)
%             if( strcmp( file_names(j).name(1:2), 'c-' ) ) 
%                 str = sprintf('%s %d', [subdir_names(i).name, '/', file_names(j).name], 1 );
%                 fprintf(fp, '%s\n', str );
%             end;
%         end;
%     end;
% end;
% 
% for i = 1:length(subdir_names)
%     if( ~strcmp(subdir_names(i).name, '.') && ~strcmp(subdir_names(i).name, '.') && subdir_names(i).isdir )
%         file_names = dir( fullfile( dir_name, subdir_names(i).name, '/*.bmp' ) );
%         for j = 1:length(file_names)
%             if( strcmp( file_names(j).name(1:2), 'cf' ) )
%                 str = sprintf('%s %d', [subdir_names(i).name, '/', file_names(j).name], 2 );
%                 fprintf(fp, '%s\n', str );
%             end;
%         end;
%     end;
% end;
% 
% fclose( fp );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% dir_name = '/home/anjan/Workspace/SignatureVerification/Datasets/GPDS300/';
% subdir_names = dir(dir_name);
% 
% fp = fopen( 'GPDS300_pairs.txt','w');
% 
% for i = 1:length(subdir_names)
%     if( ~strcmp(subdir_names(i).name, '.') && ~strcmp(subdir_names(i).name, '.') && subdir_names(i).isdir )
%         file_names = dir( fullfile( dir_name, subdir_names(i).name, '/*.bmp' ) );
%         for j = 1:length(file_names)
%             for k = j+1:length(file_names)
%                 if( strcmp( file_names(j).name(1:2), 'c-' ) && strcmp( file_names(k).name(1:2), 'c-' ) )
%                     str = sprintf( '%s %s %d', [subdir_names(i).name, '/', file_names(j).name], [subdir_names(i).name, '/', file_names(k).name], 1 );
%                     fprintf( fp, '%s\n', str );
%                 elseif( strcmp( file_names(j).name(1:2), 'c-' ) && strcmp( file_names(k).name(1:2), 'cf' ) )
%                     str = sprintf( '%s %s %d', [subdir_names(i).name, '/', file_names(j).name], [subdir_names(i).name, '/', file_names(k).name], 0 );
%                     fprintf( fp, '%s\n', str );
%                 end;
%             end;
%         end;
%     end;
% end;
% 
% fclose( fp );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% dir_name = '/home/adutta/Workspace/SignatureVerification/Datasets/GPDS960/';
% subdir_names = dir(dir_name);
% 
% fp = fopen( 'GPDS960.txt','w');
% 
% for i = 1:length(subdir_names)
%     if( ~strcmp(subdir_names(i).name, '.') && ~strcmp(subdir_names(i).name, '.') && subdir_names(i).isdir )
%         file_names = dir( fullfile( dir_name, subdir_names(i).name, '/*.jpg' ) );
%         for j = 1:length(file_names)
%             if( strcmp( file_names(j).name(1:2), 'c-' ) ) 
%                 str = sprintf('%s %d', [subdir_names(i).name, '/', file_names(j).name], 1 );
%                 fprintf(fp, '%s\n', str );
%             end;
%         end;
%     end;
% end;
% 
% for i = 1:length(subdir_names)
%     if( ~strcmp(subdir_names(i).name, '.') && ~strcmp(subdir_names(i).name, '.') && subdir_names(i).isdir )
%         file_names = dir( fullfile( dir_name, subdir_names(i).name, '/*.bmp' ) );
%         for j = 1:length(file_names)
%             if( strcmp( file_names(j).name(1:2), 'cf' ) )
%                 str = sprintf('%s %d', [subdir_names(i).name, '/', file_names(j).name], 2 );
%                 fprintf(fp, '%s\n', str );
%             end;
%         end;
%     end;
% end;
% 
% fclose( fp );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% dir_name = '/home/anjan/Workspace/SignatureVerification/Datasets/GPDS960/';
% subdir_names = dir(dir_name);
% fp = fopen( 'GPDS960_pairs.txt','w');
% 
% for i = 1:length(subdir_names)
%     if( ~strcmp(subdir_names(i).name, '.') && ~strcmp(subdir_names(i).name, '..') && subdir_names(i).isdir )
%         file_names = dir( fullfile( dir_name, subdir_names(i).name, '/*.jpg' ) );
%         for j = 1:length(file_names)
%             for k = j+1:length(file_names)                
%                 if( strcmp( file_names(j).name(1:2), 'c-' ) && strcmp( file_names(k).name(1:2), 'c-' ) ) 
%                     str = sprintf( '%s %s %d', [subdir_names(i).name, '/', file_names(j).name], [subdir_names(i).name, '/', file_names(k).name], 1 );
%                     fprintf( fp, '%s\n', str );
%                 elseif( strcmp( file_names(j).name(1:2), 'c-' ) && strcmp( file_names(k).name(1:2), 'cf' ) ) 
%                     str = sprintf( '%s %s %d', [subdir_names(i).name, '/', file_names(j).name], [subdir_names(i).name, '/', file_names(k).name], 0 );
%                     fprintf( fp, '%s\n', str );
%                 end;
%             end;
%         end;
%     end;
% end;
% 
% fclose( fp );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% dir_name = '/home/anjan/Workspace/SignatureVerification/Datasets/GPDS960';
% subdir_names = dir(dir_name);
% fp1 = fopen( fullfile( dir_name, 'list.genuine'), 'w' );
% fp2 = fopen( fullfile( dir_name, 'list.forgery'), 'w' );
% 
% for i = 1:length(subdir_names)
%     if( ~strcmp(subdir_names(i).name, '.') && ~strcmp(subdir_names(i).name, '..') && subdir_names(i).isdir )
%         file_names = dir( fullfile( dir_name, subdir_names(i).name, '/*.jpg' ) );
%         if( length( file_names ) ~= 54 )
%             keyboard;
%         end;
%         for j = 1:length(file_names)
%             parts = strsplit( file_names(j).name, '-' );            
%             if( parts{4} == 'G' )
%                 fprintf(fp1, '%s\n', fullfile( subdir_names(i).name, file_names(j).name ) );
%             elseif( parts{4} == 'F' )
%                 fprintf(fp2, '%s\n', fullfile( subdir_names(i).name, file_names(j).name ) );
%             end;            
%         end;
%     end;
% end;
% 
% fclose( fp1 );
% fclose( fp2 );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dir_name = '/home/anjan/Workspace/SignatureVerification/Datasets/GPDS960';
subdir_names = dir(dir_name);
fp1 = fopen( fullfile( dir_name, 'list.genuine'), 'w' );
fp2 = fopen( fullfile( dir_name, 'list.forgery'), 'w' );

for i = 1:length(subdir_names)
    if( ~strcmp(subdir_names(i).name, '.') && ~strcmp(subdir_names(i).name, '..') && subdir_names(i).isdir )
        file_names = dir( fullfile( dir_name, subdir_names(i).name, '/*.jpg' ) );
        if( length( file_names ) ~= 54 )
            keyboard;
        end;
        for j = 1:length(file_names)
            parts = strsplit( file_names(j).name, '-' );            
            if( strcmp( parts{1}, 'c' ) )
                fprintf(fp1, '%s\n', fullfile( subdir_names(i).name, file_names(j).name ) );
            elseif( strcmp( parts{1},'cf' ) )
                fprintf(fp2, '%s\n', fullfile( subdir_names(i).name, file_names(j).name ) );
            end;            
        end;
    end;
end;

fclose( fp1 );
fclose( fp2 );