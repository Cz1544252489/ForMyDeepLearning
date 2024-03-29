function I = ImgGenViaOpenAI(description,n,api_key)
%% This code is made by chenzhuo to generate picture
%  via openai api in 2022/12/9
%%  Paramter
% description is a string or characters list
% n is a integer and gets a big image with n-by-n sub images(1024*1024)
% And with the limitation of API, n is less than 4
% api_key gets from the webpage openai.com
%% Input test
if ~isstring(description)&&~ischar(description) , error("wrong description!"); end
if ~isinteger(uint8(n))||(isinteger(uint8(n))&&(n<1||n>=4)), error("wrong n!"); end
if ~isstring(api_key)&&~ischar(api_key) , error("wrong api_key!"); end
%% python environment test
pe = pyenv;% need a python environment
if pe.Version == ""
    error("Python not installed");
else
    disp("Python Environment is ok");
end
%% MATLAB version test
verstr=version("-release");
if (string(verstr(3:4))=="19"&&verstr(5)=='b')  || double(string(verstr(3:4)))>19.0
    disp("MATLAB version is ok");
else
    error("Update MATLAB first, older than or equal to release 2019b!");
end
%%
if pe.Status=="Loaded"
    pyrun("import importlib");
    pyrun("import openai"); % load openai function
    pyrun("openai.api_key = api_key",api_key=api_key); % need created API key
end
if pyrun("S=(importlib.util.find_spec('openai') is not None)","S")
pyrun("response=openai.Image.create(prompt=description,n=n,size='1024x1024')",...
    description=description,n=uint8(n^2));
else
    error("wrong");
end

url = string;
Image = cell(n,n);
for i=1:n^2
    url(i)=string(pyrun("image_url = response['data'][i-1]['url']","image_url",i=uint8(i)));
    Image{i}=webread(url(i));
end
I = comImage(Image);

figure;
imshow(I);
title(strcat(description," generated by Chenzhuo via OpenAI in ",string(datetime)),"Interpreter","latex");

end

function I = comImage(Image)
if size(Image,1)~=size(Image,2)
    error("wrong input sizes");
end
n=size(Image,1);
I=[];
for i = 1:n
    tmp=[];
    for j = 1:n
        tmp = [tmp,Image{i,j}];
    end
    I=[I;tmp];
end
end
