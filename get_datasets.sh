mkdir Datasets

wget https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip
unzip Geolife\ Trajectories\ 1.3.zip
mv Geolife\ Trajectories\ 1.3 Datasets
rm Geolife\ Trajectories\ 1.3.zip

wget https://3ai2lq.dm.files.1drv.com/y4mu-dEAQXcpWObZZFf86hSbCGLIBotS3N-mljQ-urBntUmyWgbL4UygnkOkc9NtcnOzXLD40FvKwW3mYknkJdaV0EsdzQCTcBe48EsBrRnBThq6EdymQciOVslMPCEBXbkvGnC2x617e9sWc6RAfU4ZUKY6eKW7iVKg1a46sMf-IvKFN3nlplpNqJk76Z5jyuQ -O tdrive.zip
unzip tdrive.zip
mv release T-Drive
mv T-Drive Datasets
rm tdrive.zip