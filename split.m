%imgPath = {'D:/MPEG_Sequece/8iVFBv2/8iVFBv2/queen/queen/','D:/MPEG_Sequece/8iVFBv2/8iVFBv2/longdress/Ply/','D:\MPEG_Sequece\phil9\ply\','D:\Record3dData\bear\','D:\Record3dData\cat\','D:\Record3dData\cup\','D:\Record3dData\frog\','D:\Record3dData\green_basket\','D:\Record3dData\kettle\','D:\Record3dData\wallet\','D:\Record3dData\water_bottle\','D:\Record3dData\nail_clipper_box\'}
%TrainPath={'D:/MPEG_Sequece/2048_data/queen/','D:/MPEG_Sequece/2048_data/longdress/','D:/MPEG_Sequece/2048_data/phil9/','D:/MPEG_Sequece/2048_data/bear/','D:/MPEG_Sequece/2048_data/cat/','D:/MPEG_Sequece/2048_data/cup/','D:/MPEG_Sequece/2048_data/frog/','D:/MPEG_Sequece/2048_data/green_basket/','D:/MPEG_Sequece/2048_data/kettle/','D:/MPEG_Sequece/2048_data/wallet/','D:/MPEG_Sequece/2048_data/water_bottle/','D:/MPEG_Sequece/2048_data/nail_clipper_box/'}
imgPath={'G:\PointCloudAttributeCompressionArtifactsReomoval\Dataset\Training\QP51\rec\'}
TrainPath={'G:\PointCloudAttributeCompressionArtifactsReomoval\Dataset\Training\Split_ply\QP51\'}
for p=1:1
    inputPath=imgPath{p};
    imgDir  = dir([inputPath '*.ply']);
    for k = 1:length(imgDir)
        frame0 = pcread([inputPath imgDir(k).name]);
        depth=length(dec2bin(floor(frame0.Count/2048))=='1')-find(dec2bin(floor(frame0.Count/2048))=='1');
        length_depth=length(depth);
        for i=1:length_depth
            if(i==1)
                last=0;
                next=last+2^depth(i)*2048;
            else
                last=next;
                next=last+2^depth(i)*2048;
            end
            newFrame_Location=frame0.Location(last+1:next,:);
            newFrame_Color=frame0.Color(last+1:next,:);
            point_cloud_split=pointCloud(newFrame_Location,'Color',newFrame_Color)
            if(depth(i)>0)
                [indx, leafs, mbrs]=buildVisualWordList(point_cloud_split.Location, depth(i));
                plyname=strsplit(imgDir(k).name,'.');
                for j=1:2^depth(i)
                    leaf_loc=point_cloud_split.Location(leafs{j},:)
                    leaf_col=point_cloud_split.Color(leafs{j},:)
                    leaf_point_cloud=pointCloud(leaf_loc,'Color',leaf_col)
                    ply_2048_name=strcat(TrainPath{p},plyname{1},'_',num2str(i),'_depth_',num2str(depth(i)),'_leaf_',num2str(j),'.',plyname{2})
                    pcwrite(leaf_point_cloud,ply_2048_name)
                end
            end
            if(depth(i)==0)
                plyname=strsplit(imgDir(k).name,'.');
                for j=1:2^depth(i)
                    ply_2048_name=strcat(TrainPath{p},plyname{1},'_',num2str(i),'_depth_',num2str(depth(i)),'_leaf_',num2str(j),'.',plyname{2})
                    pcwrite(point_cloud_split,ply_2048_name)
                end
            end
        end
    end
end




%bt_depth = 6;
%visualizeKdNode(leaf_point_cloud, leafs)





