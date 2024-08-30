-- Read the docs: https://www.lunarvim.org/docs/configuration
-- Video Tutorials: https://www.youtube.com/watch?v=sFA9kX-Ud_c&list=PLhoH5vyxr6QqGu0i7tt_XoVK9v-KvZ3m6
-- Forum: https://www.reddit.com/r/lunarvim/
-- Discord: https://discord.com/invite/Xb9B4Ny
lvim.builtin.treesitter.ensure_installed = {
  "python",
  "make",
  "lua",
}

local formatters = require "lvim.lsp.null-ls.formatters"
formatters.setup { { name = "black" } }

local linters = require "lvim.lsp.null-ls.linters"
linters.setup {
  { command = "flake8",
    args = { "--ignore=E203,E501" },
    filetypes = { "python" },
  }
}

lvim.plugins = {
  "AckslD/swenv.nvim",
  {
    "lervag/vimtex",
    lazy = false,
    init = function()
      vim.g.vimtex_view_method = "skim"
    end
  },
  { "catppuccin/nvim", name = "catppuccin", priority=1000 },
  { "rose-pine/neovim", name = "rose-pine", priority=1000 },
  "atelierbram/Base2Tone-nvim",
  "stevearc/dressing.nvim",
  "mfussenegger/nvim-dap-python",
  { "christoomey/vim-tmux-navigator", lazy = false },
  {
    "zbirenbaum/copilot.lua",
    cmd = "Copilot",
    event = "InsertEnter",
  }
}

lvim.colorscheme = "rose-pine"
vim.opt.termguicolors = true


require('swenv').setup({
  post_set_venv = function()
    vim.cmd("LspRestart")
  end,
  get_venvs = function(venvs_path)
    return require('swenv').get_venvs(venvs_path)
  end,
  venvs_path = vim.fn.expand('~/.conda/envs')
})

lvim.builtin.which_key.mappings["C"] = {
  name = "Python",
  c = { "<cmd>lua require('swenv.api').pick_venv()<cr>", "Choose Env"},
}
lvim.builtin.terminal.open_mapping = "<c-t>"

lvim.builtin.dap.active = true
local mason_path = vim.fn.glob(vim.fn.stdpath "data" .. "/mason/")
require("dap-python").setup(mason_path .. "packages/debugpy/venv/bin/python")
require("copilot").setup({
  suggestion = {
    auto_trigger = true,
    keymap = {
      accept = "<C-l>",
    }
  },
  filetypes = { yaml = true },
})
