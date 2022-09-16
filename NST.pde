import java.util.*;

ImageMap[] input(PImage img, int H, int W){
  ImageMap[] rgb = new ImageMap[3];
  img.loadPixels();
  for(int i=0; i<3; i++)
    rgb[i] = new ImageMap(H,W);
    
  for(int I=0; I<H; I++)
      for(int J=0; J<W; J++){
        rgb[0].set(I,J,red(img.pixels[W*I+J]));
        rgb[1].set(I,J,green(img.pixels[W*I+J]));
        rgb[2].set(I,J,blue(img.pixels[W*I+J]));
      }
  img.updatePixels();
  return rgb;
}

void show(ImageMap map, int a, int b){
  PImage img = createImage(map.H, map.W, RGB);
  img.loadPixels();
  for(int i=0; i<map.H; i++)
    for(int j=0; j<map.W; j++)
      img.pixels[map.W*i+j] = color(map.get(i,j), map.get(i,j), map.get(i,j));
  img.updatePixels();
  image(img, a, b);
}

void show(ImageMap[] RGBmap, int a, int b){
  println("I can show this");
  int H = RGBmap[0].H, W = RGBmap[0].W;
  PImage img = createImage(H, W, RGB);
  img.loadPixels();
  for(int i=0; i<H; i++)
    for(int j=0; j<W; j++)
      img.pixels[W*i+j] = color(RGBmap[0].get(i,j), RGBmap[1].get(i,j), RGBmap[2].get(i,j));
  img.updatePixels();
  image(img, a, b);
}

Network contentNet;
Network styleNet;
Network mainNet;
ImageMap[] Content;
ImageMap[] Style;
ImageMap[] Noise;
ArrayList<Integer> LsIndex;
ArrayList<Float> LsWeight;
int LcIndex;
float alpha, beta;
float lambda;
Adam adam;
int iter=0;


void setup(){
  size(1024, 1024);
  background(150);
  LsIndex = new ArrayList(Arrays.asList(1, 6, 11, 20, 29));
  LsWeight = new ArrayList(Arrays.asList(0.75, 0.5, 0.25, 0.25, 0.25));
  adam = new Adam();
  LcIndex = 22;
  alpha = 1.0;
  beta = 1.0;
  
  Content = input(loadImage("images/content2.jpg"), 256, 256);
  Style = input(loadImage("images/style2.jpg"), 256, 256);
  Noise = input(loadImage("images/content2.jpg"), 256, 256);
  contentNet = new VGG19();
  styleNet = new VGG19();
  mainNet = new VGG19(contentNet, styleNet);
  contentNet.forward(Content);
  styleNet.forward(Style);
  mainNet.forward(Noise);
}

void draw(){
  if(iter==0) mainNet.forward(Noise);
  else mainNet.forward();
  adam.timeplus();
  mainNet.backward();
  println(iter + "iteration");
  show(mainNet.image, 256*(iter%4), 256*((iter%16)/4));
  iter++;
}
