FROM pytorch/pytorch

RUN apt-get update
RUN apt-get upgrade -y

COPY ./apt.txt ./apt.txt
ARG DEBIAN_FRONTEND=noninteractive
RUN xargs -a apt.txt apt-get install -y --no-install-recommends && rm -rf /var/cache/*
RUN chsh -s /bin/zsh

WORKDIR "/root/"

RUN curl https://sh.rustup.rs -sSf | zsh -s -- -y
RUN . "$HOME/.cargo/env"

RUN curl --proto '=https' --tlsv1.2 -sSf https://setup.atuin.sh | zsh  -s -- -y
# RUN atuin init zsh >> /.zshrc
# RUN atuin import auto

RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# RUN mkdir -p /miniconda3
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
# RUN bash /miniconda3/miniconda.sh -b -p /miniconda3 
# RUN rm -rf /miniconda3/miniconda.sh
# ENV PATH="/miniconda3/bin:${PATH}"

RUN curl -L0 https://github.com/neovim/neovim/releases/latest/download/nvim-linux64.tar.gz | tar -xz
RUN mkdir .local
#RUN tar -xzf nvim-linux64.tar.gz
RUN mv nvim-linux64/* .local
RUN echo "PATH=/root/.local/bin:$PATH" >> .zshrc
RUN LV_BRANCH='release-1.4/neovim-0.9' curl -s https://raw.githubusercontent.com/LunarVim/LunarVim/release-1.4/neovim-0.9/utils/installer/install.sh | zsh -s -- -y
COPY config.lua /root/.config/lvim/config.lua
ENV PATH="/root/.local/bin:${PATH}"
SHELL ["/usr/bin/zsh", "-c"]
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash -s -- -y
COPY .zshrc /root/.zshrc
RUN source ~/.zshrc && nvm install node
RUN lvim --headless "+Lazy! sync" +qa
RUN lvim --headless "+Lazy! load mason.nvim +MasonInstall ruff pyright" +qa
RUN git config --global credential.helper store

RUN git config --global user.email "coleman.c.haley@gmail.com"
RUN git config --global user.name "Coleman Haley"
RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
RUN git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
RUN atuin init zsh >> /.zshrc
RUN git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
COPY .p10k.zsh /root/.p10k.zsh
#COPY requirements.txt requirements.txt
# RUN python -m pip install -r requirements.txt


